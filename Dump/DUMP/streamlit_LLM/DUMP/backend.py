from flask import Flask, request, jsonify
import psycopg2
import os
from dotenv import load_dotenv
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

app = Flask(__name__)

# Load environment variables from credentials.env
load_dotenv('credentials.env')

# Function to establish connection with PostgreSQL database using psycopg2
def connect_to_db(host: str, port: str, user: str, passwd: str, db_name: str):
    print(f"Connecting to {host} on port {port} as user {user} to database {db_name}")  # Debug print
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


# Updated function to create SQL chain with quoted column names
def create_sql_chain(conn, target_table, question):
    schema_info = get_db_schema(conn, target_table)
    quoted_schema_info = []
    
    for col in schema_info:
        if col in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:  # Remove double quotes here
            quoted_schema_info.append(f'"{col}"')  # Enclose column names in double quotes
        else:
            quoted_schema_info.append(col)
            
    # Updated template to specify using double quotes around columns A to G
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
    print(f"Generated SQL Chain: {chain}")
    return chain


def execute_query(conn, query):
    try:
        print(f"Executing Query: {query}")
        cursor = conn.cursor()
        cursor.execute(query)
        result = cursor.fetchall()
        cursor.close()
        return result
    except Exception as e:
        print(f"Query Execution Error: {str(e)}")
        return f"Error: {str(e)}"

# Function to create natural language response based on SQL query results
def create_nlp_answer(sql_query, results, question):  # Add question parameter here
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
        RunnablePassthrough(assignments={"sql_query": sql_query, "results": results_str, "question": question})  # Pass question here
        | prompt
        | llm
        | StrOutputParser()
    )

@app.route('/query', methods=['POST'])
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

        # Log the generated SQL query for debugging
        print(f"Generated SQL Query: {sql_query}")

        results = execute_query(conn, sql_query)  # Fixed function name
        if not results:
            print("No results returned by the query")
        else:
            print(f"Query Results: {results}")

        nlp_chain = create_nlp_answer(sql_query, results, question)  # Pass question here
        nlp_response = nlp_chain.invoke({})

        conn.close()

        return jsonify({'response': nlp_response}), 200
    except Exception as e:
        print(f"Error: {str(e)}")  # Log error message
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)