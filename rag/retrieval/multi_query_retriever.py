from langchain_classic.retrievers import MultiQueryRetriever
from ai.ai_client import ai_client
from rag.retrieval.add_documents import NAMESPACE
from stores.db_vector import vector_store

base_retriever = vector_store.get_store(
    NAMESPACE
).as_retriever(search_kwargs={"k": 3})
multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=ai_client.client,
    include_original=True,
)

documents = multi_query_retriever.invoke(
    "Как расторгнуть договор и что нужно оплатить?"
)
print("Найдно документов:",len(documents))
for document in documents:
    print("Источник:", document.metadata.get("source"))
    print("Раздел:", document.metadata.get("section"))
    print("Текст:", document.page_content)
    print()