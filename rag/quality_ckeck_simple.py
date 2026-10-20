from stores.db_vector import vector_store

test_cases = [
    {
        "question": "Сколько дорических храмов в Паестуме?",
        "expected_text": "три",
    },
    {
        "question": "Кто дал городу название Паестум?",
        "expected_text": "римляне",
    },
]
retriever = vector_store.as_retriever("contract_rules", search_kwargs={"k": 3})

for case in test_cases:
    documents = retriever.invoke(case["question"])

    combined_text = " ".join(
        doc.page_content.lower()
        for doc in documents
    )

    found = case["expected_text"] in combined_text

    print("Вопрос:", case["question"])
    print("Ожидаемое слово найдено:", found)
    print("Найденные фрагменты:")
    for doc in documents:
        print("-", doc.page_content)
    print()
