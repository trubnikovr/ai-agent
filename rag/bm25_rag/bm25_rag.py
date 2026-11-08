import re

from rank_bm25 import BM25Okapi
from ai.ai_client import ai_client


# Небольшая база знаний для примера.
DOCUMENTS = [
    {
        "title": "MessagesState",
        "content": (
            "MessagesState — это схема state в LangGraph, которая хранит "
            "историю сообщений. Новые сообщения добавляются в messages."
        ),
    },
    {
        "title": "Checkpoints",
        "content": (
            "Checkpointer сохраняет state LangGraph. С помощью thread_id "
            "можно продолжить диалог и восстановить историю сообщений."
        ),
    },
    {
        "title": "Tool calling",
        "content": (
            "При tool calling модель выбирает tool и формирует аргументы. "
            "Приложение выполняет tool и возвращает результат модели."
        ),
    },
    {
        "title": "Guardrails",
        "content": (
            "Guardrails проверяют запросы или ответы agent и помогают "
            "ограничивать нежелательное поведение."
        ),
    },
]


def tokenize(text: str) -> list[str]:
    """Разбить текст на слова для BM25."""
    return re.findall(r"\w+", text.casefold())


# Индексируем документы по словам, без embeddings.
tokenized_documents = [
    tokenize(f"{doc['title']} {doc['content']}")
    for doc in DOCUMENTS
]

bm25 = BM25Okapi(tokenized_documents)


def retrieve(query: str, limit: int = 2) -> list[dict]:
    """Найти документы с наибольшим BM25 score."""
    scores = bm25.get_scores(tokenize(query))

    ranked = sorted(
        zip(DOCUMENTS, scores),
        key=lambda item: item[1],
        reverse=True,
    )

    # Не передаём документы, если поиск не нашёл совпадений.
    return [
        {**document, "score": float(score)}
        for document, score in ranked
        if score > 0
    ][:limit]


def answer_question(question: str) -> None:
    documents = retrieve(question)

    if not documents:
        print("Подходящие документы не найдены.")
        return

    print("\n--- Retrieved documents ---")

    for doc in documents:
        print(f"{doc['title']}: score={doc['score']:.3f}")

    context = "\n\n".join(
        f"Название: {doc['title']}\nСодержание: {doc['content']}"
        for doc in documents
    )

    prompt = f"""
 Ты отвечаешь на вопросы о LangGraph.

 Используй только контекст ниже. Если контекста недостаточно,
 скажи об этом. Не добавляй факты, которых в нём нет.

 Контекст:
 {context}

 Вопрос:
 {question}
 """

    response = ai_client.ask(prompt)

    print("\n--- Answer ---")
    print(response)


if __name__ == "__main__":
    question = input("Вопрос: ").strip()

    if question:
        answer_question(question)
