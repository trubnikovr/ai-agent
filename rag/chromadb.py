import chromadb

chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="tourism")

collection.add(
    documents=[
        "Paestum известен хорошо сохранившимися греческими храмами.",
        "Город был основан греческими колонистами.",
        "В Паестуме находятся три дорических храма.",
    ],
    metadatas=[
        {"source": "britannica.com/place/Paestum"},
        {"source": "britannica.com/place/Paestum"},
        {"source": "britannica.com/place/Paestum"},
    ],
    ids=["paestum-1", "paestum-2", "paestum-3"],
)
results = collection.query(
    query_texts=["Сколько дорических храмов находится в Паестуме?"],
    n_results=3,
)

for document, distance in zip(
    results["documents"][0],
    results["distances"][0],
):
    print(f"Расстояние: {distance:.3f}")
    print("Фрагмент:", document)
    print()