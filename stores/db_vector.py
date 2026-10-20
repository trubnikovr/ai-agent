from langchain_core.documents import Document
from langchain_postgres import PGVector
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from config import settings

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview",
    google_api_key=settings.gemini_api_key,
)


class VectorStore:
    """PGVector facade that keeps collection selection explicit per operation."""

    def __init__(self, *,embeddings, connection: str):
        self._embeddings = embeddings
        self._connection = connection
        self._stores: dict[str, PGVector] = {}

    def get_store(self, collection_name: str) -> PGVector:
        if not collection_name:
            raise ValueError("collection_name обязателен")

        if collection_name not in self._stores:
            self._stores[collection_name] = PGVector(
                embeddings=self._embeddings,
                collection_name=collection_name,
                connection=self._connection,
                use_jsonb=True,
            )

        return self._stores[collection_name]

    def add_documents(
        self,
        collection_name: str,
        documents: list[Document],
        **kwargs,
    ) -> list[str]:
        return self.get_store(collection_name).add_documents(documents, **kwargs)

    def add(
        self,
        collection_name: str,
        text: str,
        metadata: dict | None = None,
    ) -> str:
        document = Document(page_content=text, metadata=metadata or {})
        ids = self.add_documents(collection_name, [document])
        return ids[0]

    def similarity_search(
        self,
        collection_name: str,
        query: str,
        k: int = 4,
        **kwargs,
    ) -> list[Document]:
        return self.get_store(collection_name).similarity_search(query, k=k, **kwargs)

    def similarity_search_with_score(
        self,
        collection_name: str,
        query: str,
        k: int = 4,
        **kwargs,
    ) -> list[tuple[Document, float]]:
        return self.get_store(collection_name).similarity_search_with_score(
            query, k=k, **kwargs
        )

    def as_retriever(self, collection_name: str, **kwargs):
        return self.get_store(collection_name).as_retriever(**kwargs)

def create_vector_store(embeddings, connection: str) -> VectorStore:
    return VectorStore(embeddings=embeddings, connection=connection)


vector_store = create_vector_store(
    embeddings=embeddings,
    connection=settings.pgvector_url,
)
