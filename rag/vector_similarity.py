from langchain_core.prompts import ChatPromptTemplate

from ai.ai_client import ai_client
from db_vector import vector_store

question = "Кто переименовал Посейдонию в Паестум?"

# Поиск в PostgreSQL/pgvector
documents = vector_store.similarity_search(question, k=2)

# Собираем тексты найденных фрагментов в одну строку
context = "\n\n".join(
    document.page_content
    for document in documents
)

prompt = ChatPromptTemplate.from_messages([
    ("system", """
<Instructions>
Ответь на вопрос, используя только контекст.
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
answer = ai_client.client.invoke(messages)

print(answer.content)