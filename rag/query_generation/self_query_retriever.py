from langchain_classic.retrievers import SelfQueryRetriever
from langchain_classic.retrievers.self_query.base import AttributeInfo
from langchain_postgres.translator import PGVectorTranslator
from ai.ai_client import ai_client
from rag.query_generation.seed_contracts_to_vector import NAMESPACE
from stores.db_vector import vector_store


pgvector = vector_store.get_store(NAMESPACE)

metadata_field_info = [
    AttributeInfo(
        name="contract_id",
        description="Идентификатор договора, например contract_001",
        type="string",
    ),
    AttributeInfo(
        name="section",
        description=(
            "Название раздела договора: Цена, Сроки, Приёмка, "
            "Ответственность или Расторжение"
        ),
        type="string",
    ),
    AttributeInfo(
        name="source",
        description="Имя файла, из которого извлечён фрагмент",
        type="string",
    ),
]

self_query_retriever = SelfQueryRetriever.from_llm(
    llm=ai_client.client,
    vectorstore=pgvector,
    document_contents="Текст одного раздела договора",
    metadata_field_info=metadata_field_info,
    structured_query_translator=PGVectorTranslator(),
    search_kwargs={"k": 3},
)



if __name__ == "__main__":


    question = (
        "В договоре contract_001 в разделе Цена "
        "может ли исполнитель изменить стоимость?"
    )

    structured_query = self_query_retriever.query_constructor.invoke({"query": question})
    print(structured_query)

    documents = self_query_retriever.invoke(question)
    print("Найдено с фильтром:", len(documents))

    for doc in documents:
        print(doc.metadata)
        print(doc.page_content)
