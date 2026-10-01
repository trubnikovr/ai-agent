from db_vector import vector_store

results = vector_store.similarity_search_with_score(
    query="Может ли исполнитель увеличить стоимость?",
    k=2,
)

for document, score in results:
    print("Текст:", document.page_content)
    print("Metadata:", document.metadata)
    print("Score:", score)
    print()