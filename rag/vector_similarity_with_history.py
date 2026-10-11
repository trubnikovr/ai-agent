from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from ai.ai_client import ai_client
from db_vector import vector_store

from pydantic import BaseModel, Field

class Answer(BaseModel):
    text: str = Field(description="Ответ на вопрос по найденным фрагментам")
    sources: list[str] = Field(description="Источники использованных фрагментов")

structured_llm = ai_client.client.with_structured_output(Answer)


chat_history = [
    HumanMessage(content="Расскажи о Паестуме."),
    AIMessage(content="Паестум — древний город в Южной Италии."),
]


question = "А кто его основал?"



# Поиск в PostgreSQL/pgvector
documents = vector_store.similarity_search(question, k=2)

# Собираем тексты найденных фрагментов в одну строку
# Передаём модели текст каждого фрагмента и его источник
context = "\n\n".join(
    f"Источник: {document.metadata.get('source', 'не указан')}\n"
    f"Фрагмент: {document.page_content}"
    for document in documents
)

prompt = ChatPromptTemplate.from_messages([
    ("system", """
Отвечай на вопрос по контексту.
Если ответа в контексте нет, скажи, что не знаешь.

Контекст:
{context}
"""),
    MessagesPlaceholder("chat_history"),
    ("human", "{question}"),
])

messages = prompt.invoke({
    "context": context,  # найденные фрагменты из vector_store
    "question": question,
    "chat_history": chat_history,
})

# Отправляем подготовленные сообщения Gemini
answer = structured_llm.invoke(messages)

print(answer)