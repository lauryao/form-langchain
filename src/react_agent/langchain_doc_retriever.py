from langchain_chroma import Chroma


from chromadb.utils import embedding_functions
class LangChainChromaEmbeddings:
    def __init__(self):
        self.embd=embedding_functions.DefaultEmbeddingFunction()
    def embed_query(self, query):
        return self.embd([query])[0]
    def embed_documents(self,docs):
        return self.embd(docs)
embeddings=LangChainChromaEmbeddings()    

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings
)

from langchain_core.documents import Document
documents=[
    Document("LangChain invokes LLMs"),
    Document("LangGraph runs agents")
]
vector_store.add_documents(documents)
langchain_doc_retriever=vector_store.as_retriever(search_kwargs={
    "k":1
})


# python ./src/langchain_doc_retriever.py
if __name__ == "__main__":
    retriever = langchain_doc_retriever
    res_langchain = retriever.invoke("What is LangChain?")[0]
    res_langgraph = retriever.invoke("What is LangGraph?")[0]
    print(f"Expects LangChain document, got {res_langchain}")
    print(f"Expects LangGraph document, got {res_langgraph}")