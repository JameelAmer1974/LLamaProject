
from doc_embeddings import get_doc_embedding
from postgres_connection import  cosine_similarity_search
from summary import get_document_summary_by_content

if __name__ == '__main__':

    # Query text
    query_text = (" مواجهات بين الجيش الإسرائيلي قرب نابلس خلال احتجاجات فلسطينية على المستوطنات")
    query_embedding = get_doc_embedding(query_text)
    results = cosine_similarity_search(query_embedding)
    top_docs = []
    for id, text, similarity in results:
        print(f"📌 Id: {id} | Match: {text} | 🔥 Similarity: {similarity:.4f}")
        print(f"***** اهم النقاط ******")
        print(get_document_summary_by_content(text))
        print(f"-----------------------------------")
        top_docs.append(text)



