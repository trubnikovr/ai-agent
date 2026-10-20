from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter

from stores.db_sql import SQLStore, sql_store
from stores.db_vector import VectorStore, vector_store

NAMESPACE = "contract_rules_parents"


class ParentDocumentRetrieverManager:
    """Build and cache ParentDocumentRetriever instances from the configured stores."""

    def __init__(self, *, vector_store: VectorStore, sql_store: SQLStore):
        self._vector_store = vector_store
        self._sql_store = sql_store
        self._retrievers: dict[str, ParentDocumentRetriever] = {}

    def get_retriever(
        self,
        *,
        namespace: str = NAMESPACE,
        parent_splitter: RecursiveCharacterTextSplitter = 0,
        child_splitter: RecursiveCharacterTextSplitter = 0,
        **kwargs,
    ) -> ParentDocumentRetriever:
        if not namespace:
            raise ValueError("namespace обязателен")

        if namespace not in self._retrievers:
            self._retrievers[namespace] = ParentDocumentRetriever(
                vectorstore=self._vector_store.get_store(namespace),
                docstore=self._sql_store.get_kv_docstore(namespace=namespace),
                parent_splitter=parent_splitter or RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=100,
                ),
                child_splitter=child_splitter or RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=100,
                ),
                **kwargs,
            )

        return self._retrievers[namespace]


def create_parent_document_retriever_manager(
    vector_store: VectorStore,
    sql_store: SQLStore,
) -> ParentDocumentRetrieverManager:
    return ParentDocumentRetrieverManager(vector_store=vector_store, sql_store=sql_store)


parent_document_retriever_manager = create_parent_document_retriever_manager(
    vector_store=vector_store,
    sql_store=sql_store,
)
