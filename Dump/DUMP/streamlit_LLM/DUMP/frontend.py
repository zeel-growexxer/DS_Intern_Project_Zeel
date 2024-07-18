import streamlit as st
from flask import Flask, request, jsonify
from flask_cors import CORS
import psycopg2
import requests
import threading
import os
from dotenv import load_dotenv
import pandas as pd
import plotly.express as px
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

# Load environment variables from credentials.env
load_dotenv('credentials.env')

# Create Flask app
flask_app = Flask(__name__)
CORS(flask_app)

# Function to establish connection with PostgreSQL database using psycopg2
def connect_to_db(host, port, user, passwd, db_name):
    conn = psycopg2.connect(
        host=host,
        port=port,
        user=user,
        password=passwd,
        database=db_name
    )
    return conn

# Function to fetch table and column information from the database schema
def get_db_schema(conn, table_name):
    cursor = conn.cursor()
    cursor.execute(f"""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema='public' AND table_name='{table_name}'
    """)
    schema_info = [row[0] for row in cursor.fetchall()]
    cursor.close()
    return schema_info

# Function to create SQL chain with quoted column names
def create_sql_chain(conn, target_table, question):
    schema_info = get_db_schema(conn, target_table)
    quoted_schema_info = []
    
    for col in schema_info:
        if col in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
            quoted_schema_info.append(f'"{col}"')
        else:
            quoted_schema_info.append(col)
            
    template = f"""
        Based on the table schema of table '{target_table}', write a SQL query to answer the question.
        If the question is on the dataset, the SQL query must take all the columns into consideration.
        Only provide the SQL query, without any additional text or characters.
        Ensure the query includes the table name {target_table} in the FROM clause.
        If user asks a question related to correlation between 2 columns and if either column is non-numeric, display an appropriate message instead of calculating the correlation.
    
        Table schema: {quoted_schema_info}
        Question: {question}

        SQL Query:
    """
    prompt = ChatPromptTemplate.from_template(template=template)
    llm = ChatGroq(model="llama3-8b-8192", temperature=0.2, groq_api_key=os.getenv('InfoQuest_API_KEY'))

    chain = (
        RunnablePassthrough(assignments={"schema": quoted_schema_info, "question": question})
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

# Function to execute SQL query
def execute_query(conn, query):
    try:
        cursor = conn.cursor()
        cursor.execute(query)
        result = cursor.fetchall()
        cursor.close()
        return result
    except Exception as e:
        return f"Error: {str(e)}"

# Function to create natural language response based on SQL query results
def create_nlp_answer(sql_query, results, question):
    results_str = "\n".join([str(row) for row in results])

    template = f"""
        Based on the results of the SQL query '{sql_query}', write a natural language response.
        Consider the initial {question} while generating the output in natural language.
        Do not write "Based on the SQL query results" or "Therefore, the natural language response would be:" in the response.

        Query Results:
        {results_str}
    """
    prompt = ChatPromptTemplate.from_template(template=template)
    llm = ChatGroq(model="llama3-8b-8192", temperature=0.2, groq_api_key=os.getenv('InfoQuest_API_KEY'))

    return (
        RunnablePassthrough(assignments={"sql_query": sql_query, "results": results_str, "question": question})
        | prompt
        | llm
        | StrOutputParser()
    )

# Flask route to handle query
@flask_app.route('/query', methods=['POST'])
def handle_query():
    data = request.get_json()
    question = data.get('question')

    if not question:
        return jsonify({'error': 'Question not provided'}), 400

    hostname = os.getenv('DB_HOST')
    port = os.getenv('DB_PORT')
    username = os.getenv('DB_USER')
    password = os.getenv('DB_PASSWORD')
    database_name = os.getenv('DB_NAME')
    target_table = os.getenv('TABLE_NAME')

    try:
        conn = connect_to_db(hostname, port, username, password, database_name)
        schema_info = get_db_schema(conn, target_table)
        sql_chain = create_sql_chain(conn, target_table, question)
        sql_query_response = sql_chain.invoke({})
        sql_query = sql_query_response.strip()

        results = execute_query(conn, sql_query)

        nlp_chain = create_nlp_answer(sql_query, results, question)
        nlp_response = nlp_chain.invoke({})

        conn.close()

        return jsonify({'response': nlp_response, 'results': results}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Start Flask server in a separate thread
def start_flask_app():
    flask_app.run(debug=True, use_reloader=False)

threading.Thread(target=start_flask_app).start()

# Streamlit app
def send_query(question):
    url = 'http://127.0.0.1:5000/query'
    payload = {'question': question}
    headers = {'Content-Type': 'application/json'}

    try:
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            return response.json()
        else:
            return {'error': f"Error: {response.status_code}"}
    except requests.exceptions.RequestException as e:
        return {'error': f"Error: {str(e)}"}

def main():
    st.title("InfoQuest")
    st.markdown(
        """
        <style>
        body {
            background-color: offwhite;
        }
        .css-1yywi0x {
            background-color: #4CAF50;
            color: white;
            border-color: #4CAF50;
        }
        .message-container {
            margin-bottom: 10px;
        }
        .user-message {
            background-color: #e0f7fa;
            padding: 10px;
            border-radius: 10px;
            text-align: right;
            color: black;
        }
        .bot-message {
            background-color: #e3f2fd;
            padding: 10px;
            border-radius: 10px;
            text-align: left;
            color: black;
        }
        .label {
            font-weight: bold;
            margin-bottom: 5px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    if 'history' not in st.session_state:
        st.session_state.history = []

    question = st.text_input("Type your question here:", key="input_question")

    if st.button("Send"):
        if question:
            response_data = send_query(question)
            response = response_data.get('response')
            results = response_data.get('results')

            st.session_state.history.append((question, "User"))
            st.session_state.history.append((response, "Bot"))

            if results:
                # Display visualization if results are available
                df = pd.DataFrame(results, columns=[col[0] for col in results])
                fig = px.line(df)  # You can customize the visualization as needed
                st.plotly_chart(fig)

    for i, (msg, sender) in enumerate(st.session_state.history):
        if sender == "User":
            st.markdown('<div class="label">You:</div><div class="message-container"><div class="user-message">{}</div></div>'.format(msg), unsafe_allow_html=True)
        elif sender == "Bot":
            st.markdown('<div class="label">Bot:</div><div class="message-container"><div class="bot-message">{}</div></div>'.format(msg), unsafe_allow_html=True)

if __name__ == "__main__":
    main()




# import streamlit as st
# import requests

# # Function to send query to backend and retrieve response
# def send_query(question):
#     url = 'http://127.0.0.1:5000/query'  # Update if deploying elsewhere
#     payload = {'question': question}
#     headers = {'Content-Type': 'application/json'}

#     try:
#         response = requests.post(url, json=payload, headers=headers)
#         if response.status_code == 200:
#             return response.json().get('response')
#         else:
#             return f"Error: {response.status_code}"
#     except requests.exceptions.RequestException as e:
#         return f"Error: {str(e)}"

# # Main function to define Streamlit UI
# def main():
#     st.title("InfoQuest")
#     st.markdown(
#         """
#         <style>
#         /* Styling for the background */
#         body {
#             background-color: offwhite;
#         }
#         /* Styling for the Send button */
#         .css-1yywi0x {
#             background-color: #4CAF50;
#             color: white;
#             border-color: #4CAF50;
#         }
#         /* Styling for user and bot messages */
#         .message-container {
#             margin-bottom: 10px;
#         }
#         .user-message {
#             background-color: #e0f7fa; /* Light blue background for user messages */
#             padding: 10px;
#             border-radius: 10px;
#             text-align: right;
#             color: black; /* Black text color */
#         }
#         .bot-message {
#             background-color: #e3f2fd; /* Light blue background for bot messages */
#             padding: 10px;
#             border-radius: 10px;
#             text-align: left;
#             color: black; /* Black text color */
#         }
#         .label {
#             font-weight: bold;
#             margin-bottom: 5px;
#         }
#         </style>
#         """,
#         unsafe_allow_html=True
#     )

#     # Initialize session state to preserve chat history
#     if 'history' not in st.session_state:
#         st.session_state.history = []

#     # Input field for user question
#     question = st.text_input("Type your question here:", key="input_question")

#     # Button to submit the question
#     if st.button("Send"):
#         if question:
#             # Send query to backend and get response
#             response = send_query(question)

#             # Append question and response to history
#             st.session_state.history.append((question, "User"))
#             st.session_state.history.append((response, "Bot"))

#     # Display chat history
#     for i, (msg, sender) in enumerate(st.session_state.history):
#         if sender == "User":
#             st.markdown('<div class="label">You:</div><div class="message-container"><div class="user-message">{}</div></div>'.format(msg), unsafe_allow_html=True)
#         elif sender == "Bot":
#             st.markdown('<div class="label">Bot:</div><div class="message-container"><div class="bot-message">{}</div></div>'.format(msg), unsafe_allow_html=True)

# if __name__ == "__main__":
#     main()