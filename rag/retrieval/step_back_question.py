from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from ai.ai_client import ai_client
from config import configure_langchain
from rag.retrieval.add_documents import NAMESPACE
from stores.db_vector import vector_store

configure_langchain()

# Та же коллекция, куда добавили тестовые документы.
pgvector = vector_store.get_store(NAMESPACE)
retriever = pgvector.as_retriever(search_kwargs={"k": 3})

# 1. Просим Gemini сделать шаг назад и сформулировать общий поисковый запрос.
step_back_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
Сформулируй один общий вопрос, ответ на который поможет ответить
на исходный вопрос. Не отвечай на него и не добавляй фактов.
Верни только формулировку вопроса.
""",
    ),
    ("human", "{question}"),
])

step_back_chain = step_back_prompt | ai_client.client | StrOutputParser()

# 2. Ищем документы в pgvector по общему вопросу.
def format_documents(documents):
    return "\n\n".join(
        f"Источник: {doc.metadata.get('source', 'не указан')}\n"
        f"Раздел: {doc.metadata.get('section', 'не указан')}\n"
        f"{doc.page_content}"
        for doc in documents
    )

# 3. Просим Gemini ответить на первоначальный вопрос
#    по фрагментам, найденным через общий запрос.
answer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
Ответь на исходный вопрос, используя только контекст.
Если контекст не содержит ответа, так и скажи.
Не добавляй фактов, которых нет в контексте.

Контекст:
{context}
""",
    ),
    (
        "human",
        """
Исходный вопрос: {question}

Общий поисковый вопрос: {step_back_question}
""",
    ),
])

answer_chain = answer_prompt | ai_client.client | StrOutputParser()

question = "Что будет, если исполнитель задержит выполнение работ?"

# Шаг 1: общий вопрос для поиска
step_back_question = step_back_chain.invoke({"question": question})
print("Общий поисковый вопрос:", step_back_question)

# Шаг 2: поиск в векторной базе именно по общему вопросу
documents = retriever.invoke(step_back_question)
context = format_documents(documents)

print("\nНайденные фрагменты:")
print(context)

# Шаг 3: ответ на исходный вопрос по найденному контексту
answer = answer_chain.invoke({
    "question": question,
    "step_back_question": step_back_question,
    "context": context,
})

print("\nОтвет:")
print(answer)