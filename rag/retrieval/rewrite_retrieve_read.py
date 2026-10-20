from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

from ai.ai_client import ai_client
from config import configure_langchain
from rag.retrieval.add_documents import NAMESPACE
from stores.db_vector import vector_store

configure_langchain()

# Укажи то же имя коллекции, по которому поиск уже нашёл запись.
pgvector = vector_store.get_store(NAMESPACE)
retriever = pgvector.as_retriever(search_kwargs={"k": 3})

chat_history = [
    HumanMessage(content="Как определяется цена услуг по договору?"),
    AIMessage(content=(
        "Цена фиксированная. Изменить её можно только "
        "по письменному соглашению сторон."
    )),
]

question = "А исполнитель может изменить её сам?"

# Rewrite: превращаем уточнение в самостоятельный запрос.
rewrite_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
Перепиши последнее сообщение пользователя в самостоятельный запрос
для поиска по договору, учитывая историю.
Верни только запрос. Не отвечай и не добавляй условий.
""",
    ),
    MessagesPlaceholder("chat_history"),
    ("human", "{question}"),
])

rewrite_chain = rewrite_prompt | ai_client.client | StrOutputParser()

search_query = rewrite_chain.invoke({
    "chat_history": chat_history,
    "question": question,
})

# Retrieve: ищем договорные фрагменты.
documents = retriever.invoke(search_query)

if not documents:
    raise ValueError(
        f"По запросу не найдены документы: {search_query}"
    )

context = "\n\n".join(
    f"Источник: {doc.metadata.get('source', 'не указан')}\n"
    f"Раздел: {doc.metadata.get('section', 'не указан')}\n"
    f"{doc.page_content}"
    for doc in documents
)

# Read: отвечаем на исходный вопрос по найденному контексту.
answer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
Ответь на вопрос только по приведённым фрагментам договора.
Не добавляй фактов и юридических норм, которых в них нет.
Если ответа в контексте нет, прямо скажи об этом.
""",
    ),
    (
        "human",
        """
Исходный вопрос: {question}

Фрагменты договора:
<context>
{context}
</context>

Дай краткий ответ и укажи раздел договора, на который опираешься.
""",
    ),
])

answer_chain = answer_prompt | ai_client.client | StrOutputParser()

answer = answer_chain.invoke({
    "question": question,
    "context": context,
})

print("Поисковый запрос:", search_query)
print("\nНайденный контекст:\n", context)
print("\nОтвет:\n", answer)