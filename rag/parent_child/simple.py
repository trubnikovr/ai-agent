from retrievers.parent_document_retriever_manager import parent_document_retriever_manager

query = "Может ли исполнитель сам изменить цену?"
results = parent_document_retriever_manager.get_retriever(namespace='contract_rules_parents').invoke(query)
 #print(results)

for i, document in enumerate(results, start=1):
    print(f"\nРезультат {i}")
    print("Текст:", document.page_content)
    print("Метаданные:", document.metadata)
