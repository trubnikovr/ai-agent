from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

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

query_llm = ai_client.client.with_structured_output(SearchQuery)

rewrite_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
Преобразуй последний вопрос пользователя в самостоятельный запрос для поиска
по документам. Используй историю, чтобы понять, к чему относятся местоимения.
Если вопрос уже самостоятельный, оставь его смысл без изменений.
Верни только поле query.
""",
    ),
    MessagesPlaceholder("chat_history"),
    ("human", "{question}"),
])



def ask(question: str) -> str:

    # 1. Уточняем вопрос для поиска с учётом истории
    query_messages = rewrite_prompt.invoke({
        "chat_history": chat_history,
        "question": question,
    })

    search_query = query_llm.invoke(query_messages).query

    documents = vector_store.similarity_search("contract_rules", search_query, k=3)

    context = "\n\n".join(
        f"Источник: {doc.metadata.get('source', 'не указан')}\n"
        f"Фрагмент: {doc.page_content}"
        for doc in documents)

    # 3. Формируем ответ по найденным фрагментам и истории
    messages = prompt.invoke({
        "context": context,
        "chat_history": chat_history,
        "question": question,
    })
    answer = structured_llm.invoke(messages)

    chat_history.extend([
        HumanMessage(content=question),
        AIMessage(content=answer.text),
    ])

    return {
        "search_query": search_query,
        "text": answer.text,
        "sources": list(dict.fromkeys(
            doc.metadata["source"]
            for doc in documents
            if doc.metadata.get("source")
        )),
    }





    # # Ищем документы по текущему вопросу
    # documents = vector_store.similarity_search("contract_rules", question, k=3)
    #
    # context = "\n\n".join(
    #     document.page_content
    #     for document in documents
    # )
    #
    # # Собираем prompt из контекста, истории и вопроса
    # messages = prompt.invoke({
    #     "context": context,
    #     "chat_history": chat_history,
    #     "question": question,
    # })
    #
    # answer = structured_llm.invoke(messages)
    #
    # chat_history.extend([
    #     HumanMessage(content=question),
    #     AIMessage(content=answer.text),
    # ])
    #
    # return answer





print(ask("Расскажи о Паестуме."))
print(ask("А кто его основал?"))
