from dotenv import load_dotenv
from pprint import pprint
import requests
import os
#from database_connection import save_document
from doc_embeddings import get_doc_embedding
#from sentence_transformers import SentenceTransformer
from postgres_connection import  cosine_similarity_search



def generate_answer(context, question):
    prompt = f"""المحتوى:\n{context}\n\nالسؤال: {question}\nالإجابة:"""
   # inputs = tokenizer(prompt, return_tensors="pt")
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=874)


    print("Input length:", inputs['input_ids'].shape[1])
    print("Model max length:", getattr(model.config, "n_positions", "unknown"))
    outputs = model.generate(**inputs, max_new_tokens=150)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)



# def generate_answer_openai(docs, question):
#     context = "\n\n".join([f"Document {i + 1}:\n{doc}" for i, doc in enumerate(docs)])
#
#     prompt = f"""You are a helpful assistant. Use the context below to answer the user's question.
#
# Context:
# \"\"\"
# {context}
# \"\"\"
#
# Question: {question}
# Answer:"""
#     response = client.chat.completions.create(
#         model= "gpt-3.5-turbo",
#         messages=[
#             {"role": "system", "content": "You answer based only on the provided context."},
#             {"role": "user", "content": prompt}
#         ],
#         temperature=0.3,
#     )
#
#     return response.choices[0].message.content.strip()

if __name__ == '__main__':

    # Query text
    query_text = (" مواجهات بين الجيش الإسرائيلي قرب نابلس خلال احتجاجات فلسطينية على المستوطنات")
    query_embedding = get_doc_embedding(query_text).tolist()
    results = cosine_similarity_search(query_embedding)
    top_docs = []
    for text, similarity in results:
        print(f"📌 Match: {text} | 🔥 Similarity: {similarity:.4f}")
        top_docs.append(text)
    from transformers import AutoTokenizer, AutoModelForCausalLM

    # tokenizer = AutoTokenizer.from_pretrained("riotu-lab/ArabianGPT-03B")
    # model = AutoModelForCausalLM.from_pretrained("riotu-lab/ArabianGPT-03B")
    # context = "\n".join(top_docs)
    # answer = generate_answer(context, query_text)
    # print(answer)

    import psycopg2
    from rasa_sdk import Action, Tracker
    from rasa_sdk.executor import CollectingDispatcher
    from typing import Any, Text, Dict, List
