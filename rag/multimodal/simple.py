from langchain_core.documents import Document

from rag.multimodal.ingest_pdf import NAMESPACE
from stores.db_vector import vector_store

pgvector = vector_store.get_store(NAMESPACE)

def search_pdf_index(query: str, k: int = 4) -> list[Document]:
    """Ищет похожие фрагменты и проверяет аргументы поиска."""

    query = query.strip()
    if not query:
        raise ValueError("Поисковый запрос не должен быть пустым")

    if k < 1 or k > 20:
        raise ValueError("k должен быть от 1 до 20")

    return pgvector.similarity_search(query, k=k)

if __name__ == "__main__":
    try:


        results = search_pdf_index(
            "Какие сроки и условия оплаты указаны в документе?",
            k=4,
        )

        print("\nРезультаты поиска:")
        for result in results:
            print(
                f"\nИсточник: {result.metadata.get('source')}, "
                f"страница: {result.metadata.get('page')}, "
                f"извлечение: {result.metadata.get('extraction_method')}"
            )
            print(result.page_content)

    except Exception as exc:
        raise SystemExit(f"Ошибка обработки: {exc}") from exc