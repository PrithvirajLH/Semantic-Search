import streamlit as st
import elasticsearch as es
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

index_name = "vector_db"
try:
    es = Elasticsearch(
        "https://localhost:9200",
        basic_auth=("elastic", "0AvRKA69Ud34+k2Uphbn"),
        verify_certs=False
    )
except ConnectionError as e:
    print("Connection Error:", e)

if es.ping():
    print("Successfully connected to ElasticSearch")
else:
    print("Oops!Elastic Search is not connected")


def search(input_keyword):
    model = SentenceTransformer("all-mpnet-base-v2")
    vector_of_input_keyword = model.encode(input_keyword)

    query = {
        "field": "vector_embedding",
        "query_vector": vector_of_input_keyword,
        "k": 10,
        "num_candidates": 1000
    }

    res = es.search(index="vector_db",
                    knn=query,
                    source=["Series_Title", "Overview"]
                    )
    results = res["hits"]["hits"]

    return results



st.title("IMDB Movie Search")

search_query = st.text_input("Enter your search query")

if st.button("Search"):
    if search_query:
        results = search(search_query)

        st.subheader("Search Results")
        for result in results:
            with st.container():
                if '_source' in result:
                    try:
                        st.header(f"{result['_source']['Series_Title']}")
                    except Exception as e:
                        print(e)
                    try:
                        st.write(f"Description: {result['_source']['Overview']}")
                    except Exception as e:
                        print(e)
                    st.divider()



