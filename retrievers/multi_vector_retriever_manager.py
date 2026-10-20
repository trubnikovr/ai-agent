from langchain_classic.retrievers import MultiVectorRetriever
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from stores.db_sql import SQLStore, sql_store
from stores.db_vector import VectorStore, vector_store


NAMESPACE = "contract_rules_parents_multi_vector"
ID_KEY = "doc_id"


class MultiVectorRetrieverDocument:
    """Создаёт MultiVectorRetriever и индексирует parent/child документы."""

    def __init__(
        self,
        *,
        vector_store: VectorStore,
        sql_store: SQLStore,
        child_splitter: RecursiveCharacterTextSplitter,
        namespace: str = NAMESPACE,
        id_key: str = ID_KEY,
        search_kwargs: dict | None = None,
    ) -> None:
        if not namespace:
            raise ValueError("namespace обязателен")

        self._namespace = namespace
        self._child_splitter = child_splitter

        # Получаем хранилища один раз и передаём их retriever.
        vectorstore = vector_store.get_store(namespace)
        docstore = sql_store.get_kv_docstore(namespace=namespace)

        self._retriever = MultiVectorRetriever(
            vectorstore=vectorstore,
            docstore=docstore,
            id_key=id_key,
            search_kwargs=search_kwargs or {"k": 3},
        )

    def get_retriever(self) -> MultiVectorRetriever:
        """Возвращает стандартный LangChain retriever для поиска."""
        return self._retriever

    def index_documents(self, documents: list[Document],
                        hypothetical_questions: list[str] = None) -> None:
        """Сохраняет родительские документы и их child-фрагменты."""

        vectorstore = self._retriever.vectorstore
        docstore = self._retriever.docstore
        id_key = self._retriever.id_key

        for parentDocument in documents:
            parent_id = self._stable_parent_id(parentDocument)
            children = self._child_splitter.split_documents([parentDocument])

            # Связываем каждый child с родительским документом.
            for child in children:
                child.metadata[id_key] = parent_id

            # Удаляем старую версию, чтобы повторная индексация
            # не оставляла прежние child-фрагменты.
            vectorstore.delete(filter={id_key: parent_id})
            docstore.mdelete([parent_id])

            # Parent хранится в SQLStore, child-фрагменты — в vector store.
            docstore.mset([(parent_id, parentDocument)])

            if hypothetical_questions is not None:
                question_documents = self._generate_hypothetical_questions(hypothetical_questions)
                for question_doc in question_documents:
                    question_doc.metadata.update({
                        "category": "contract",
                        "parent_id": parent_id,
                    })

                children = [*children, *question_documents]

            if children:
                vectorstore.add_documents(children)

            print(
                f"Индексирован раздел "
                f"«{parentDocument.metadata.get('section', parent_id)}»: "
                f"{len(children)} child-фрагментов"
            )

    @staticmethod
    def _stable_parent_id(document: Document) -> str:
        """Возвращает стабильный ID по source и section."""

        source = document.metadata.get("source")
        section = document.metadata.get("section")

        if not source or not section:
            raise ValueError(
                "Для каждого документа нужны metadata['source'] "
                "и metadata['section']"
            )

        return f"{source}:{section}"

    def _generate_hypothetical_questions(self,
                                         hypothetical_questions
                                         ):
        question_documents = [
            Document(
                page_content=question,
                metadata={

                    "representation": "hypothetical_question",
                },
            )
            for question in hypothetical_questions
        ]

        return question_documents


def create_multi_vector_retriever_manager(
    *,
    vector_store: VectorStore,
    sql_store: SQLStore,
    namespace: str = NAMESPACE,
    id_key: str = ID_KEY,
    child_splitter: RecursiveCharacterTextSplitter | None = None,
    search_kwargs: dict | None = None,
) -> MultiVectorRetrieverDocument:
    return MultiVectorRetrieverDocument(
        vector_store=vector_store,
        sql_store=sql_store,
        namespace=namespace,
        id_key=id_key,
        child_splitter=child_splitter
        or RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50,
        ),
        search_kwargs=search_kwargs
        ,
    )

multi_vector_retriever_manager = create_multi_vector_retriever_manager(
    vector_store=vector_store,
    sql_store=sql_store,
    namespace=NAMESPACE,
    id_key=ID_KEY,
    child_splitter=RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
    ),
)