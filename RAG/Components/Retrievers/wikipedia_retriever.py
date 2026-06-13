from langchain_community.retrievers import WikipediaRetriever

retriever = WikipediaRetriever(top_k_results=2, lang='en')

query = "The geopolitical history of India and Pakistan from the perspective of a Chinese person"

docs = retriever.invoke(query)

for i, doc in enumerate(docs):
    print(f"--- Document {i+1} ---")
    print(doc.page_content)
    print()