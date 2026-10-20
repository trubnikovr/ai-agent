# rag/parent_child/ask.py

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

from ai.ai_client import ai_client
from config import configure_langchain
from retrievers.parent_document_retriever_manager import parent_document_retriever_manager


configure_langchain()

# Настройки должны совпадать с индексатором:
# тот же namespace, vector store, SQL docstore и конфигурация splitter.
retriever = parent_document_retriever_manager.get_retriever(
    namespace="contract_rules_parents",
)


class ContractAnswer(BaseModel):
    answer: str = Field(
        description="Ответ на вопрос только по найденному контексту договора"
    )


prompt = ChatPromptTemplate.from_messages([
    ("system", """
Ты анализируешь текст договора.

Отвечай только на основании переданного контекста.
Не добавляй факты, которых нет в контексте.
Если контекста недостаточно для ответа, так и скажи.

<Context>
{context}
</Context>
"""),
    ("human", """
<Question>
{question}
</Question>
"""),
])

structured_llm = ai_client.client.with_structured_output(ContractAnswer)


def format_documents(documents) -> str:
    """Форматирует parent-документы и добавляет их метаданные в контекст."""
    formatted = []

    for index, document in enumerate(documents, start=1):
        source = document.metadata.get("source", "не указан")
        section = document.metadata.get("section", "не указан")

        formatted.append(
            f"[Фрагмент {index}]\n"
            f"Источник: {source}\n"
            f"Раздел: {section}\n"
            f"{document.page_content}"
        )

    return "\n\n".join(formatted)


def ask(question: str) -> dict:
    # ParentDocumentRetriever:
    # 1. ищет подходящие child-фрагменты в PGVector;
    # 2. по их ID получает parent-документы из SQLStore.
    documents = retriever.invoke(question)

    if not documents:
        return {
            "answer": "Подходящие фрагменты договора не найдены.",
            "sources": [],
        }

    context = format_documents(documents)

    messages = prompt.invoke({
        "context": context,
        "question": question,
    })

    # Gemini возвращает Pydantic-объект с полем answer.
    model_result = structured_llm.invoke(messages)

    # Источники берём из реально найденных документов,
    # а не просим Gemini генерировать их.
    sources = []
    seen_sources = set()

    for document in documents:
        source = document.metadata.get("source", "не указан")
        section = document.metadata.get("section", "не указан")
        source_key = (source, section)

        if source_key not in seen_sources:
            seen_sources.add(source_key)
            sources.append({
                "source": source,
                "section": section,
            })

    return {
        "answer": model_result.answer,
        "sources": sources,
    }


if __name__ == "__main__":
    question = "Может ли исполнитель сам увеличить стоимость услуг?"
    result = ask(question)

    print("Ответ:")
    print(result["answer"])

    print("\nИсточники:")
    for source in result["sources"]:
        print(
            f"- {source['source']} "
            f"(раздел: {source['section']})"
        )
