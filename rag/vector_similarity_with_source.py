from langchain_core.prompts import ChatPromptTemplate

from ai.ai_client import ai_client
from db_vector import vector_store

from pydantic import BaseModel, Field

class Answer(BaseModel):
    text: str = Field(description="Ответ на вопрос по найденным фрагментам")
    sources: list[str] = Field(description="Источники использованных фрагментов")

structured_llm = ai_client.client.with_structured_output(Answer)


question = "Кто переименовал Посейдонию в Паестум?"



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
<Instructions>
Ответь на вопрос, используя только контекст. Укажи так же источник
Если ответа в контексте нет, скажи, что не знаешь.
</Instructions>

<Context>
{context}
</Context>
"""),
    ("human", "<Question>{question}</Question>"),
])

messages = prompt.invoke({
    "context": context,
    "question": question,
})

# Отправляем подготовленные сообщения Gemini
answer = structured_llm.invoke(messages)

print(answer)