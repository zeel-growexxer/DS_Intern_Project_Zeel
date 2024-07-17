# import streamlit as st
# import requests
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import psycopg2
# import os
# from dotenv import load_dotenv
# import pandas as pd
# import plotly.express as px
# import plotly.graph_objects as go
# from langchain_core.runnables import RunnablePassthrough
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_groq import ChatGroq

# # Initialize Flask app
# app = Flask(__name__)
# CORS(app)

# # Load environment variables from credentials.env
# load_dotenv('credentials.env')

# # Function to establish connection with PostgreSQL database using psycopg2
# def connect_to_db():
#     return psycopg2.connect(
#         host=os.getenv('DB_HOST'),
#         port=os.getenv('DB_PORT'),
#         user=os.getenv('DB_USER'),
#         password=os.getenv('DB_PASSWORD'),
#         database=os.getenv('DB_NAME')
#     )

# # Function to fetch table and column information from the database schema
# def get_db_schema(conn, table_name):
#     cursor = conn.cursor()
#     cursor.execute(f"""
#         SELECT column_name
#         FROM information_schema.columns
#         WHERE table_schema='public' AND table_name='{table_name}'
#     """)
#     schema_info = [row[0] for row in cursor.fetchall()]
#     cursor.close()
#     return schema_info

# # Function to create SQL chain with quoted column names
# def create_sql_chain(conn, target_table, question):
#     schema_info = get_db_schema(conn, target_table)
#     quoted_schema_info = [f'"{col}"' if col in ['A', 'B', 'C', 'D', 'E', 'F', 'G'] else col for col in schema_info]

#     template = f"""
#         Based on the table schema of table '{target_table}', write a SQL query to answer the question.
#         If the question is on the dataset, the SQL query must take all the columns into consideration.
#         Only provide the SQL query, without any additional text or characters.
#         Ensure the query includes the table name {target_table} in the FROM clause.
#         If user asks a question related to correlation between 2 columns and if either column is non-numeric, display an appropriate message instead of calculating the correlation.
    
#         Table schema: {quoted_schema_info}
#         Question: {question}

#         SQL Query:
#     """
#     prompt = ChatPromptTemplate.from_template(template=template)
#     llm = ChatGroq(model="llama3-8b-8192", temperature=0.2, groq_api_key=os.getenv('InfoQuest_API_KEY'))

#     chain = (
#         RunnablePassthrough(assignments={"schema": quoted_schema_info, "question": question})
#         | prompt
#         | llm
#         | StrOutputParser()
#     )
#     return chain

# def execute_query(conn, query):
#     try:
#         cursor = conn.cursor()
#         cursor.execute(query)
#         result = cursor.fetchall()
#         cursor.close()
#         return result
#     except Exception as e:
#         return f"Error: {str(e)}"

# # Function to create natural language response based on SQL query results
# def create_nlp_answer(sql_query, results, question):
#     results_str = "\n".join([str(row) for row in results])

#     template = f"""
#         Based on the results of the SQL query '{sql_query}', write a natural language response.
#         Consider the initial {question} while generating the output in natural language.
#         Do not write "Based on the SQL query results" or "Therefore, the natural language response would be:" in the response.

#         Query Results:
#         {results_str}
#     """
#     prompt = ChatPromptTemplate.from_template(template=template)
#     llm = ChatGroq(model="llama3-8b-8192", temperature=0.2, groq_api_key=os.getenv('InfoQuest_API_KEY'))

#     return (
#         RunnablePassthrough(assignments={"sql_query": sql_query, "results": results_str, "question": question})
#         | prompt
#         | llm
#         | StrOutputParser()
#     )

# # Flask route to handle queries
# @app.route('/query', methods=['POST'])
# def handle_query():
#     data = request.get_json()
#     question = data.get('question')

#     if not question:
#         return jsonify({'error': 'Question not provided'}), 400

#     try:
#         conn = connect_to_db()
#         target_table = os.getenv('TABLE_NAME')
#         sql_chain = create_sql_chain(conn, target_table, question)
#         sql_query_response = sql_chain.invoke({})
#         sql_query = sql_query_response.strip()

#         results = execute_query(conn, sql_query)
#         nlp_chain = create_nlp_answer(sql_query, results, question)
#         nlp_response = nlp_chain.invoke({})

#         conn.close()

#         return jsonify({'response': nlp_response, 'results': results}), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# # Streamlit app
# def send_query(question):
#     url = 'http://127.0.0.1:5000/query'
#     payload = {'question': question}
#     headers = {'Content-Type': 'application/json'}

#     try:
#         response = requests.post(url, json=payload, headers=headers)
#         if response.status_code == 200:
#             return response.json()
#         else:
#             return f"Error: {response.status_code}"
#     except requests.exceptions.RequestException as e:
#         return f"Error: {str(e)}"

# def main():
#     st.title("InfoQuest")
#     st.markdown(
#         """
#         <style>
#         body { background-color: offwhite; }
#         .css-1yywi0x { background-color: #4CAF50; color: white; border-color: #4CAF50; }
#         .message-container { margin-bottom: 10px; }
#         .user-message { background-color: #e0f7fa; padding: 10px; border-radius: 10px; text-align: right; color: black; }
#         .bot-message { background-color: #e3f2fd; padding: 10px; border-radius: 10px; text-align: left; color: black; }
#         .label { font-weight: bold; margin-bottom: 5px; }
#         </style>
#         """,
#         unsafe_allow_html=True
#     )

#     if 'history' not in st.session_state:
#         st.session_state.history = []

#     question = st.text_input("Type your question here:", key="input_question")

#     if st.button("Send"):
#         if question:
#             response = send_query(question)
#             st.session_state.history.append((question, "User"))
#             if isinstance(response, dict):
#                 st.session_state.history.append((response.get('response'), "Bot"))
#                 results = response.get('results')
#                 if results:
#                     df = pd.DataFrame(results)
#                     st.session_state.history.append(("Generating visualization...", "Bot"))
#                     generate_visualization(df, question)
#             else:
#                 st.session_state.history.append((response, "Bot"))

#     for i, (msg, sender) in enumerate(st.session_state.history):
#         if sender == "User":
#             st.markdown(f'<div class="label">You:</div><div class="message-container"><div class="user-message">{msg}</div></div>', unsafe_allow_html=True)
#         elif sender == "Bot":
#             st.markdown(f'<div class="label">Bot:</div><div class="message-container"><div class="bot-message">{msg}</div></div>', unsafe_allow_html=True)


# def generate_visualization(df, question):
#     visualized = False  # Flag to track if a visualization has been generated

#     for col in df.columns:
#         if not visualized:  # Only generate one visualization per question
#             if df[col].dtype == 'object':
#                 if df[col].nunique() >= 10:
#                     # Bar plot for categorical data with more than 10 unique values
#                     fig = px.bar(df, x=col, title=f"Bar plot of {col}")
#                     fig.update_layout(xaxis_title=col, yaxis_title="Count", showlegend=True)
#                     st.plotly_chart(fig)
#                     visualized = True
#                 else:
#                     # Pie chart for discrete categorical data
#                     fig = px.pie(df, names=col, title=f"Pie chart of {col}")
#                     st.plotly_chart(fig)
#                     visualized = True
#             elif df[col].dtype in ['int64', 'float64']:
#                 if df[col].nunique() > 10:
#                     # Histogram for distribution and counts of numerical data
#                     fig = px.histogram(df, x=col, title=f"Histogram of {col}", nbins=20)
#                     fig.update_layout(xaxis_title=col, yaxis_title="Count", showlegend=True)
#                     st.plotly_chart(fig)
#                     visualized = True
#                 else:
#                     # Box plot for checking spread and outliers of numerical data
#                     fig = px.box(df, y=col, title=f"Box plot of {col}")
#                     fig.update_layout(xaxis_title=col, yaxis_title="Value", showlegend=True)
#                     st.plotly_chart(fig)
#                     visualized = True
#             else:
#                 # Scatter plot for exploring relationships between two numerical variables
#                 fig = px.scatter(df, x=col, y=question, title=f"Scatter plot of {col} vs {question}")
#                 fig.update_layout(xaxis_title=col, yaxis_title=question, showlegend=True)
#                 st.plotly_chart(fig)
#                 visualized = True

#     if not visualized:
#         st.write("No appropriate visualization found for this question.")

        
# if __name__ == "__main__":
#     main()
#     app.run(debug=True, use_reloader=False)
