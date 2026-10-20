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


def ask(question: str) -> str:
    # Ищем документы по текущему вопросу
    documents = vector_store.similarity_search("contract_rules", question, k=3)

    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    # Собираем prompt из контекста, истории и вопроса
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

    return answer





print(ask("Расскажи о Паестуме."))
print(ask("А кто его основал?"))
