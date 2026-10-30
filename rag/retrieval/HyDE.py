from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from ai.ai_client import ai_client
from config import configure_langchain
from rag.retrieval.add_documents import NAMESPACE
from stores.db_vector import vector_store

configure_langchain()

# Используй namespace той коллекции, где уже нашёлся пункт о цене.
pgvector = vector_store.get_store(NAMESPACE)
retriever = pgvector.as_retriever(search_kwargs={"k": 3})

question = "Может ли исполнитель самостоятельно изменить цену услуг?"

# 1. Создаём гипотетический текст для поиска.
hyde_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
Напиши короткий гипотетический фрагмент договора,
который мог бы отвечать на вопрос пользователя.
Не представляй его как реальный текст договора.
Верни только фрагмент.
""",
    ),
    ("human", "{question}"),
])

hyde_chain = hyde_prompt | ai_client.client | StrOutputParser()
hypothetical_document = hyde_chain.invoke({"question": question})

print("Гипотетический фрагмент для поиска:")
print(hypothetical_document)

# 2. Ищем реальные документы по гипотетическому тексту.
documents = retriever.invoke(hypothetical_document)

print(f"\nНайдено реальных фрагментов: {len(documents)}")

for document in documents:
    print("\nИсточник:", document.metadata.get("source", "не указан"))
    print("Раздел:", document.metadata.get("section", "не указан"))
    print(document.page_content)

