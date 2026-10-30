from langchain_classic.retrievers import MultiQueryRetriever

# pgvector и search_query уже созданы предыдущим этапом:
# pgvector = vector_store.get_store("ИМЯ_ТВОЕЙ_КОЛЛЕКЦИИ")
# search_query = rewrite_chain.invoke(...)

base_retriever = pgvector.as_retriever(
    search_kwargs={"k": 2},
)

multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=ai_client.client,
    include_original=True,
)

documents = multi_query_retriever.invoke(search_query)

print(f"Найдено фрагментов: {len(documents)}")

for document in documents:
    print("\nРаздел:", document.metadata.get("section"))
    print(document.page_content)