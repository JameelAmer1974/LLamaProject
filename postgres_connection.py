import psycopg2
from dotenv import load_dotenv
# PostgreSQL connection details
DB_NAME = "SCSDB"
DB_USER = "postgres"
DB_PASSWORD = "SmartSoft@2014"
DB_HOST = "localhost"  # Change if hosted remotely
DB_PORT = "5432"       # Default PostgreSQL port
import os


def connect(db_name,db_user,db_password,db_host,db_port):
    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )
    return conn

def db_connect():
    load_dotenv()
    db_name = os.getenv("DB_NAME")
    db_user = os.getenv("DB_USER")
    db_password = os.getenv("DB_PASSWORD")
    db_host = os.getenv("DB_HOST")
    db_port = os.getenv("DB_PORT")
    return connect(db_name,db_user,db_password,db_host,db_port)

def check_connection(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT version();")
    version = cursor.fetchone()
    return version

def create_tables(conn,sql_statement):

    try:
        cursor = conn.cursor()
    # statement like
    #        CREATE TABLE IF NOT EXISTS users (
    #            id SERIAL PRIMARY KEY,
    #            name VARCHAR(100),
    #            email VARCHAR(100) UNIQUE
    #        );
    #
        cursor.execute(sql_statement)
        conn.commit()
        print("Table created successfully!")
        cursor.close()
        return "Table created successfully"
    except Exception as e:
        print("Error creating table: ", e)
        return "Error creating table"

def execute_query(conn,sql_statement):
    cursor = conn.cursor()
    cursor.execute(sql_statement)
    conn.commit()
    print("Query executed successfully!")
    cursor.close()
    return "Query executed successfully"

def fetch_data(conn,sql_statement):
    cursor = conn.cursor()
    cursor.execute(sql_statement)
    data = cursor.fetchall()
    return data

def search(sql_statement,query_embedding):
    conn = db_connect()
    cursor = conn.cursor()
    cursor.execute(sql_statement, (query_embedding,))
    return cursor.fetchall()

def cosine_similarity_search(query_embedding):
    conn = db_connect()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT text, 1 - (embedding <=> %s::vector) AS similarity
        FROM documents_new
        ORDER BY similarity DESC
        LIMIT 3;
    """, (query_embedding,))
    return cursor.fetchall()

def add_documents(conn,sentence,embedding):
    # Prepare the document
    cursor = conn.cursor()
    doc = {
        "text": sentence,
        "embedding": embedding.tolist()
    }
    # Insert into MongoDB
    cursor.execute("INSERT INTO documents_new (text, embedding) VALUES (%s, %s)", (sentence, embedding.tolist()))
    conn.commit()
    cursor.close()
    #print("Embedding saved successfully!")
    return doc


if __name__ == '__main__':
    # Connect to PostgreSQL
    conn = connect(DB_NAME,DB_USER,DB_PASSWORD,DB_HOST,DB_PORT)
    check_connection(conn)
