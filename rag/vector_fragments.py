from pydantic import BaseModel, Field

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough

from ai.ai_client import ai_client
from stores.db_vector import vector_store


# Структура финального ответа
class Answer(BaseModel):
    text: str = Field(description="Ответ на вопрос по найденным фрагментам")
    sources: list[str] = Field(description="Источники найденных фрагментов")


# Структура поискового запроса
class SearchQuery(BaseModel):
    query: str = Field(description="Самостоятельный запрос для поиска по документам")


# Модели
answer_llm = ai_client.client.with_structured_output(Answer)
query_llm = ai_client.client.with_structured_output(SearchQuery)

# Поиск в PostgreSQL/pgvector
retriever = vector_store.as_retriever("contract_rules", search_kwargs={"k": 3})


# Prompt для переформулирования вопроса с учётом истории
query_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
Преобразуй последний вопрос пользователя в самостоятельный запрос для поиска.
Используй историю, чтобы понять, к чему относятся местоимения.
Если вопрос уже самостоятельный, сохрани его смысл.
""",
    ),
    MessagesPlaceholder("chat_history"),
    ("human", "{question}"),
])


# Prompt для финального ответа
answer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
Отвечай только на основе контекста.
Если ответа в контексте нет, скажи, что не знаешь.
Не придумывай факты.

Контекст:
{context}
""",
    ),
    MessagesPlaceholder("chat_history"),
    ("human", "{question}"),
])


def format_documents(documents: list[Document]) -> str:
    """Объединить тексты найденных документов в контекст."""
    return "\n\n".join(
        f"Источник: {doc.metadata.get('source', 'не указан')}\n"
        f"Фрагмент: {doc.page_content}"
        for doc in documents
    )


def rewrite_question(inputs: dict) -> str:
    """Переформулировать вопрос с учётом истории диалога."""
    messages = query_prompt.invoke({
        "chat_history": inputs["chat_history"],
        "question": inputs["question"],
    })

    result = query_llm.invoke(messages)
    return result.query


def retrieve_documents(inputs: dict) -> list[Document]:
    """Найти документы по переформулированному запросу."""
    return retriever.invoke(inputs["search_query"])


# Собираем RAG-цепочку
rag_chain = (
    RunnablePassthrough.assign(
        search_query=rewrite_question,
    )
    .assign(
        documents=retrieve_documents,
    )
    .assign(
        context=lambda inputs: format_documents(inputs["documents"]),
    )
    | answer_prompt
    | answer_llm
)


# История текущего диалога
chat_history = []


def ask(question: str) -> Answer:
    result = rag_chain.invoke({
        "question": question,
        "chat_history": chat_history,
    })

    # Сохраняем исходный вопрос и текст ответа для следующих уточнений
    chat_history.extend([
        HumanMessage(content=question),
        AIMessage(content=result.text),
    ])

    return result


# Пример диалога
first_answer = ask("Расскажи о Паестуме.")
print("Ответ:", first_answer.text)
print("Источники:", first_answer.sources)

# second_answer = ask("А кто его основал?")
# print("Ответ:", second_answer.text)
# print("Источники:", second_answer.sources)
