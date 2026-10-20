from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough

from ai.ai_client import ai_client
from stores.db_vector import vector_store

from pydantic import BaseModel, Field


class Answer(BaseModel):
    text: str = Field(description="Ответ на вопрос по найденным фрагментам")
    sources: list[str] = Field(description="Источники использованных фрагментов")

structured_llm = ai_client.client.with_structured_output(Answer)

chat_history = []


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

class SearchQuery(BaseModel):
    query: str = Field(
        description="Самостоятельный поисковый запрос без местоимений и ссылок на прошлые реплики"
    )

retriever = vector_store.as_retriever("contract_rules", search_kwargs={"k": 3})


def format_documents(documents: list[Document]) -> str:
    return "\n\n".join(
        f"Источник: {doc.metadata.get('source', 'не указан')}\n"
        f"{doc.page_content}"
        for doc in documents
    )


query_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Переформулируй последний вопрос в самостоятельный запрос для поиска. "
        "Используй историю, чтобы понять местоимения. Верни только поле query.",
    ),
    MessagesPlaceholder("chat_history"),
    ("human", "{question}"),
])
query_llm = ai_client.client.with_structured_output(SearchQuery)

rag_chain = (
        RunnablePassthrough.assign(
            search_query=lambda inputs: query_llm.invoke(
                query_prompt.invoke({
                    "chat_history": inputs["chat_history"],
                    "question": inputs["question"],
                })
            ).query
        )
        .assign(
            context=lambda inputs: format_documents(
                retriever.invoke(inputs["search_query"])
            )
        )
    | prompt
    | structured_llm
)

result = rag_chain.invoke(
    {
    "question": "Кто переименовал Посейдонию в Паестум?",
    "chat_history": chat_history,
}
)

print(result.text)
print(result.sources)
