from dotenv import load_dotenv
from pprint import pprint
import requests
import os
import numpy as np
import pandas as pd
from datasets import Dataset
#from database_connection import save_document
from sentence_transformers import SentenceTransformer
from postgres_connection import add_documents,connect
from arabic_text_normalization import normalize_arabic_text

from langchain_community.embeddings import OllamaEmbeddings
from sqlalchemy import create_engine, text
import json

# Ollama Embedding wrapper (calls local model)
embedding_model = OllamaEmbeddings(model="llama3")

load_dotenv()



# Load the model
def get_doc_embedding(document_content="SSS Document"):
    doc_embedding = embedding_model.embed_query(document_content)
   # doc_embedding.append(embedding_model.embed_query(document_content))  # ← Now `embeddings` is a list!
    return doc_embedding


if __name__ == '__main__':
    import os
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    db_name = os.getenv("DB_NAME")
    db_user = os.getenv("DB_USER")
    db_password = os.getenv("DB_PASSWORD")
    db_host = os.getenv("DB_HOST")
    db_port = os.getenv("DB_PORT")
    conn = connect(db_name, db_user, db_password, db_host, db_port)

    bbc_data = pd.read_csv('bbc_news_arabic_summarization.csv')
    bbc_data.dropna(subset=['text', 'summary'], inplace=True)
    bbc_data = bbc_data.drop(columns=['id', 'url', 'title'])

    total_doc= len(bbc_data)
    print(total_doc)
    for i, sentence in enumerate(bbc_data["text"]):
        normalize_arabic_text(sentence)
        embeddings = get_doc_embedding(sentence)
        add_documents(conn, sentence,embeddings)

        print(i,"of:", total_doc)
