import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
import psycopg2
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

app = Flask(__name__)
CORS(app)

load_dotenv('credentials.env')

def connect_to_db():
    return psycopg2.connect(
        host=os.getenv('DB_HOST'),
        port=os.getenv('DB_PORT'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD'),
        database=os.getenv('DB_NAME')
    )

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

def create_sql_chain(conn, target_table, question):
    schema_info = get_db_schema(conn, target_table)
    quoted_schema_info = [f'"{col}"' if col in ['A', 'B', 'C', 'D', 'E', 'F', 'G'] else col for col in schema_info]

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

def execute_query(conn, query):
    try:
        cursor = conn.cursor()
        cursor.execute(query)
        result = cursor.fetchall()
        cursor.close()
        return result
    except Exception as e:
        return f"Error: {str(e)}"

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

@app.route('/query', methods=['POST'])
def handle_query():
    data = request.get_json()
    question = data.get('question')

    if not question:
        return jsonify({'error': 'Question not provided'}), 400

    try:
        conn = connect_to_db()
        target_table = os.getenv('TABLE_NAME')
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

if __name__ == "__main__":
    app.run(debug=True, threaded=False)
