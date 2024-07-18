import streamlit as st
import json
import psycopg2
import os
from dotenv import load_dotenv
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

# Load environment variables from credentials.env
load_dotenv('credentials.env')

# Function to establish connection with PostgreSQL database using psycopg2
def connect_to_db():
    try:
        conn = psycopg2.connect(
            host=os.getenv('DB_HOST'),
            port=os.getenv('DB_PORT'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD'),
            database=os.getenv('DB_NAME')
        )
        return conn
    except Exception as e:
        st.write(f"Error connecting to database: {str(e)}")
        return None

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
    quoted_schema_info = [f'"{col}"' if col in ['A', 'B', 'C', 'D', 'E', 'F', 'G'] else col for col in schema_info]

    template = f"""
        Based on the table schema of table '{target_table}', write a SQL query to answer the question.
        If the question is on the dataset, the SQL query must take all the columns into consideration.
        Only provide the SQL query, without any additional text or characters.
        Ensure the query includes the table name {target_table} in the FROM clause.

        If user asks a question related to correlation between 2 columns and if either column is non-numeric, display an appropriate message instead of calculating the correlation.
        If the question is related to the column names in the dataset, please refer the schema {quoted_schema_info} to identify the column names.

        If the question involves complex conditions or calculations, consider using SQL subqueries, CASE statements, and CTEs (Common Table Expressions) to ensure the query is efficient and clear.
        Provide subqueries for nested queries when needed to filter or aggregate data before the main query.
        Use CTEs for better readability and organization of complex queries, especially when multiple steps are involved.
        Apply CASE statements to handle conditional logic directly within the query results.

        Examples of SQL constructs to use when appropriate:
        - Subqueries: SELECT * FROM (SELECT ... FROM ...) AS subquery;
        - CTEs: WITH cte_name AS (SELECT ... FROM ...) SELECT * FROM cte_name;
        - CASE statements: SELECT CASE WHEN condition THEN result ELSE other_result END AS column_name;

        Specific columns to refer to based on the question context:
        - Coverage level: column 'A'
        - Smoking type: column 'B'
        - Car type: column 'C'
        - Purpose of the vehicle: column 'D'
        - Safety features: column 'E'
        - Driver's historic record: column 'F'
        - Area where the user will drive the car (rural, urban, suburban or hazardous): column 'G'
        - Previous car type: column 'c_previous'
        
        If there comes a question which is unrelated to the table {target_table} and is based on general knowledge, answer the question based on your knowledge rather than throwing errors.
        If the question is about the dataset description, refer the schema {quoted_schema_info} and table {target_table} to understand the column names and the data.
         
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

# Function to execute SQL query and fetch results
def execute_query(conn, query):
    try:
        cursor = conn.cursor()
        cursor.execute(query)
        colnames = [desc[0] for desc in cursor.description]  # Get column names
        result = cursor.fetchall()
        cursor.close()
        return colnames, result
    except Exception as e:
        return None, f"Error: {str(e)}"

# Function to create natural language response based on SQL query results
def create_nlp_answer(sql_query, results, question):
    results_str = "\n".join([str(row) for row in results])

    template = f"""
        Based on the results of the SQL query '{sql_query}', write a natural language response.
        Consider the initial {question} while generating the output in natural language.
        Do not write "Based on the SQL query results" or "Therefore, the natural language response would be:" in the response.
        If you are unable to solve very complex questions or it is not possible to get answers based on SQL queries then please answer politely rather than giving errors.

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

# Function to send query to PostgreSQL database and retrieve response
def send_query(question, history):
    try:
        conn = connect_to_db()
        if conn:
            target_table = os.getenv('TABLE_NAME')
            sql_chain = create_sql_chain(conn, target_table, question)
            sql_query_response = sql_chain.invoke({})
            sql_query = sql_query_response.strip()

            colnames, results = execute_query(conn, sql_query)
            if colnames and results:
                nlp_chain = create_nlp_answer(sql_query, results, question)
                nlp_response = nlp_chain.invoke({})

                history.append((question, "User"))
                history.append((nlp_response, "Bot"))

                save_session_history(history)
                conn.close()

                return {'response': nlp_response, 'results': {'colnames': colnames, 'data': results}}
            else:
                conn.close()
                return {'error': 'No results found.'}
        else:
            return {'error': "Failed to connect to database."}
    except Exception as e:
        return {'error': str(e)}

# Function to save session history
def save_session_history(history):
    with open('session_history.json', 'w') as f:
        json.dump(history, f)

# Function to load session history
def load_session_history():
    if os.path.exists('session_history.json'):
        with open('session_history.json', 'r') as f:
            return json.load(f)
    return []

# Main function to run the Streamlit app
def main():
    st.title("InfoQuest")
    st.markdown(
        """
        <style>
        body { background-color: offwhite; }
        .message-container { margin-bottom: 10px; }
        .user-message { background-color: #EBECF0; padding: 10px; border-radius: 10px; text-align: right; color: black; }
        .bot-message { background-color: #EBECF0; padding: 10px; border-radius: 10px; text-align: left; color: black; }
        .label { font-weight: bold; margin-bottom: 5px; }
        </style>
        """,
        unsafe_allow_html=True
    )

    if 'session_history' not in st.session_state:
        st.session_state.session_history = []

    history = load_session_history()
    selected_question = None

    st.sidebar.title("Chat History")
    questions = [msg for msg, sender in history if sender == "User"]

    for i, question in enumerate(questions):
        if st.sidebar.button(question, key=f"history_{i}"):
            selected_question = i

    if selected_question is not None:
        selected_history = history[selected_question*2:]  # Each user message has a bot response, so step by 2
        for i, (msg, sender) in enumerate(selected_history):
            if sender == "User":
                st.markdown(f'<div class="label">You:</div><div class="message-container"><div class="user-message">{msg}</div></div>', unsafe_allow_html=True)
            elif sender == "Bot":
                st.markdown(f'<div class="label">Bot:</div><div class="message-container"><div class="bot-message">{msg}</div></div>', unsafe_allow_html=True)

    question = st.text_input("Type your question here:")

    if st.button("Send"):
        if question:
            response = send_query(question, st.session_state.session_history)
            if isinstance(response, dict):
                if 'error' in response:
                    st.error(f"{response['error']}")
                else:
                    st.markdown(f'<div class="label">You:</div><div class="message-container"><div class="user-message">{question}</div></div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="label">Bot:</div><div class="message-container"><div class="bot-message">{response.get("response")}</div></div>', unsafe_allow_html=True)

    # Display all chat history
    for i, (msg, sender) in enumerate(st.session_state.session_history):
        if sender == "User":
            st.markdown(f'<div class="label">You:</div><div class="message-container"><div class="user-message">{msg}</div></div>', unsafe_allow_html=True)
        elif sender == "Bot":
            st.markdown(f'<div class="label">Bot:</div><div class="message-container"><div class="bot-message">{msg}</div></div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()


# import streamlit as st
# import json
# import psycopg2
# import os
# from dotenv import load_dotenv
# from langchain_core.runnables import RunnablePassthrough
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_groq import ChatGroq

# # Load environment variables from credentials.env
# load_dotenv('credentials.env')

# # Function to establish connection with PostgreSQL database using psycopg2
# def connect_to_db():
#     try:
#         conn = psycopg2.connect(
#             host=os.getenv('DB_HOST'),
#             port=os.getenv('DB_PORT'),
#             user=os.getenv('DB_USER'),
#             password=os.getenv('DB_PASSWORD'),
#             database=os.getenv('DB_NAME')
#         )
#         return conn
#     except Exception as e:
#         st.write(f"Error connecting to database: {str(e)}")
#         return None

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
#         If the question is related to the column names in the dataset, please refer the schema {quoted_schema_info} to identify the column names.

#         If the question involves complex conditions or calculations, consider using SQL subqueries, CASE statements, and CTEs (Common Table Expressions) to ensure the query is efficient and clear.
#         Provide subqueries for nested queries when needed to filter or aggregate data before the main query.
#         Use CTEs for better readability and organization of complex queries, especially when multiple steps are involved.
#         Apply CASE statements to handle conditional logic directly within the query results.

#         Examples of SQL constructs to use when appropriate:
#         - Subqueries: SELECT * FROM (SELECT ... FROM ...) AS subquery;
#         - CTEs: WITH cte_name AS (SELECT ... FROM ...) SELECT * FROM cte_name;
#         - CASE statements: SELECT CASE WHEN condition THEN result ELSE other_result END AS column_name;

#         Specific columns to refer to based on the question context:
#         - Coverage level: column 'A'
#         - Smoking type: column 'B'
#         - Car type: column 'C'
#         - Purpose of the vehicle: column 'D'
#         - Safety features: column 'E'
#         - Driver's historic record: column 'F'
#         - Area where the user will drive the car (rural, urban, suburban or hazardous): column 'G'
#         - Previous car type: column 'c_previous'

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

# # Function to execute SQL query and fetch results
# def execute_query(conn, query):
#     try:
#         cursor = conn.cursor()
#         cursor.execute(query)
#         colnames = [desc[0] for desc in cursor.description]  # Get column names
#         result = cursor.fetchall()
#         cursor.close()
#         return colnames, result
#     except Exception as e:
#         return None, f"Error: {str(e)}"

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

# # Function to send query to PostgreSQL database and retrieve response
# def send_query(question, history):
#     try:
#         conn = connect_to_db()
#         if conn:
#             target_table = os.getenv('TABLE_NAME')
#             sql_chain = create_sql_chain(conn, target_table, question)
#             sql_query_response = sql_chain.invoke({})
#             sql_query = sql_query_response.strip()

#             colnames, results = execute_query(conn, sql_query)
#             if colnames and results:
#                 nlp_chain = create_nlp_answer(sql_query, results, question)
#                 nlp_response = nlp_chain.invoke({})

#                 history.append((question, "User"))
#                 history.append((nlp_response, "Bot"))

#                 save_session_history(history)
#                 conn.close()

#                 return {'response': nlp_response, 'results': {'colnames': colnames, 'data': results}}
#             else:
#                 conn.close()
#                 return {'error': 'No results found or error executing query.'}
#         else:
#             return {'error': "Failed to connect to database."}
#     except Exception as e:
#         return {'error': str(e)}

# # Function to save session history
# def save_session_history(history):
#     with open('session_history.json', 'w') as f:
#         json.dump(history, f)

# # Function to load session history
# def load_session_history():
#     if os.path.exists('session_history.json'):
#         with open('session_history.json', 'r') as f:
#             return json.load(f)
#     return []

# # Main function to run the Streamlit app
# def main():
#     st.title("InfoQuest")
#     st.markdown(
#         """
#         <style>
#         body { background-color: offwhite; }
#         .message-container { margin-bottom: 10px; }
#         .user-message { background-color: #EBECF0; padding: 10px; border-radius: 10px; text-align: right; color: black; }
#         .bot-message { background-color: #EBECF0; padding: 10px; border-radius: 10px; text-align: left; color: black; }
#         .label { font-weight: bold; margin-bottom: 5px; }
#         </style>
#         """,
#         unsafe_allow_html=True
#     )

#     history = load_session_history()
#     selected_question = None

#     st.sidebar.title("Chat History")
#     questions = [msg for msg, sender in history if sender == "User"]

#     for i, question in enumerate(questions):
#         if st.sidebar.button(question, key=f"history_{i}"):
#             selected_question = i

#     if selected_question is not None:
#         selected_history = history[selected_question*2:]  # Each user message has a bot response, so step by 2
#         for i, (msg, sender) in enumerate(selected_history):
#             if sender == "User":
#                 st.markdown(f'<div class="label">You:</div><div class="message-container"><div class="user-message">{msg}</div></div>', unsafe_allow_html=True)
#             elif sender == "Bot":
#                 st.markdown(f'<div class="label">Bot:</div><div class="message-container"><div class="bot-message">{msg}</div></div>', unsafe_allow_html=True)

#     question = st.text_input("Type your question here:")

#     if st.button("Send"):
#         if question:
#             response = send_query(question, history)
#             if isinstance(response, dict):
#                 if 'error' in response:
#                     st.error(f"{response['error']}")
#                 else:
#                     st.markdown(f'<div class="label">You:</div><div class="message-container"><div class="user-message">{question}</div></div>', unsafe_allow_html=True)
#                     st.markdown(f'<div class="label">Bot:</div><div class="message-container"><div class="bot-message">{response.get("response")}</div></div>', unsafe_allow_html=True)

#     # Display all chat history
#     for i, (msg, sender) in enumerate(history):
#         if sender == "User":
#             st.markdown(f'<div class="label">You:</div><div class="message-container"><div class="user-message">{msg}</div></div>', unsafe_allow_html=True)
#         elif sender == "Bot":
#             st.markdown(f'<div class="label">Bot:</div><div class="message-container"><div class="bot-message">{msg}</div></div>', unsafe_allow_html=True)

# if __name__ == "__main__":
#     main()
    
    
    
    
    
    
    
    
# template = f"""
    #     Based on the table schema of table '{target_table}', write a SQL query to answer the question.
    #     If the question is on the dataset, the SQL query must take all the columns into consideration.
    #     Only provide the SQL query, without any additional text or characters.
    #     Ensure the query includes the table name {target_table} in the FROM clause.
    #     If user asks a question related to correlation between 2 columns and if either column is non-numeric, display an appropriate message instead of calculating the correlation.
    #     If the question is related to the column names in the dataset, please refer the schema {quoted_schema_info} to identify the column names.
    #     If the question is related to coverage level, use column 'A' from the table {target_table}.
    #     If the question is related to smoking type, use column 'B' from the table {target_table}.
    #     If the question is related to car type, use column 'C' from the table {target_table}.
    #     If the question is related to the purpose of the vehicle, use column 'D' from the table {target_table}.
    #     If the question is related to safety features, use column 'E' from the table {target_table}.
    #     If the question is related to driver's historic record, use column 'F' from the table {target_table}.
    #     If the question is related to the area where the user will drive the car (rural, urban, suburban or haardous), use column 'G' from the table {target_table}.
    #     If the question is related to the previous car type, use column 'c_previous' from the table {target_table}.
    
    #     Table schema: {quoted_schema_info}
    #     Question: {question}

    #     SQL Query:
    # """