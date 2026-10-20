from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from stores.db_vector import vector_store
from ai.ai_client import ai_client


question = "Кто переименовал Посейдонию в Паестум?"

results = vector_store.similarity_search("contract_rules", question, k=2)

for document in results:
    print("Источник:", document.metadata.get("source"))
    print("Текст:", document.page_content)
    print()


# 5. Настроить retriever и prompt
retriever = vector_store.as_retriever(
    "contract_rules",
    search_kwargs={"k": 2},
)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
<Instructions>
Отвечай на вопрос только на основе контекста.
Если ответа в контексте нет, скажи, что не знаешь.
Не придумывай факты.
</Instructions>

<Context>
{context}
</Context>
""",
    ),
    (
        "human",
        """
<Question>
{question}
</Question>
""",
    ),
])


# 6. Собрать RAG-цепочку
rag_chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough(),
    }
    | prompt
    | ai_client.client
)


# 7. Задать вопрос
answer = rag_chain.invoke(question)

print("Ответ:")
print(answer.content)
