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
load_dotenv()



# Load the model
def get_doc_embedding(document_content="SSS Document"):
    model_name = os.getenv("model_name")
    #model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
    model = SentenceTransformer(model_name)
    # Get embeddings
    embeddings = model.encode(document_content)
    return embeddings


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

    # Randomly sample 50% of the dataset
    #bbc_data = full_bbc_data.sample(frac=0.5, random_state=42)
    bbc_data.dropna(subset=['text', 'summary'], inplace=True)
    bbc_data = bbc_data.drop(columns=['id', 'url', 'title'])
    #bbc_data = bbc_data.sample(frac=0.1, random_state=42)
    #bbc_dataset = Dataset.from_pandas(bbc_data)
    # Get embeddings
    #embeddings = get_doc_embedding(sentences)
    total_doc= len(bbc_data)
    print(total_doc)
    # Print embeddings
    for i, sentence in enumerate(bbc_data["text"]):
        normalize_arabic_text(sentence)
        embeddings = get_doc_embedding(sentence)
        add_documents(conn, sentence,embeddings)

        print(i,"of:", total_doc)
        #print(f"Embedding: {embeddings[:5]}...")  # Showing first 5 values for brevity
