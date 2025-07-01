from dotenv import load_dotenv
from postgres_connection import  db_connect,fetch_data
import ollama
load_dotenv()

def get_summary(sql_query="Select id,text from documents_llama where id=36136"):

    conn=db_connect()
    data=fetch_data(conn,sql_query)
    content = "\n".join([f"رقم المقال : {id}, النص: {text}" for id, text in data])
    prompt = (
        "أجب فقط باللغة العربية، ولا تستخدم أي لغة أخرى. "
        "استخراج اهم النقاط من  المقالة التالية:\n\n"
        f"{content}\n\n"
        "الرجاء استخراج اهم النقاط من النص."
    )
    # Query LLaMA
    response = ollama.chat(model='llama3', messages=[
        {"role": "system", "content": "أنت مساعد ذكي يجيب فقط باللغة العربية، لا تستخدم الإنجليزية أبداً."},
        {"role": "user", "content": prompt}
    ])
    summary= response['message']['content']
    return summary

def get_document_summary_by_content(document_content="SSS Document"):
    content = document_content
    prompt = (
        "أجب فقط باللغة العربية، ولا تستخدم أي لغة أخرى. "
        "استخراج اهم النقاط من  المقالة التالية:\n\n"
        f"{content}\n\n"
        "الرجاء استخراج اهم النقاط من النص."
    )
    # Query LLaMA
    response = ollama.chat(model='llama3', messages=[
        {"role": "system", "content": "أنت مساعد ذكي يجيب فقط باللغة العربية، لا تستخدم الإنجليزية أبداً."},
        {"role": "user", "content": prompt}
    ])
    summary= response['message']['content']
    return summary

if __name__ == '__main__':
    sql_query ="Select id,text from documents_llama where id=29016"
    summary = get_summary(sql_query)
    print(summary)


