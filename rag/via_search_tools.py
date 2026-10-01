from langchain.agents import create_agent
from langchain_core.tools import tool

from ai.ai_client import ai_client
from db_vector import vector_store


@tool
def search_contract_rules(query: str) -> str:
    """
    Ищет информацию во внутренней базе договорных правил.

    Используй для вопросов об оплате, стоимости,
    неустойке и условиях договора.
    """

    results = vector_store.similarity_search(
        query=query,
        k=3,
    )

    if not results:
        return "Подходящие документы не найдены"

    parts = []

    for document in results:
        section = document.metadata.get(
            "section",
            "unknown",
        )

        parts.append(
            f"Раздел: {section}\n"
            f"Текст: {document.page_content}"
        )

    return "\n\n".join(parts)


agent = create_agent(
    model=ai_client.client,
    tools=[search_contract_rules],
    system_prompt="""
Ты помощник по анализу договоров.

Правила:
- перед ответом по условиям договора используй
  search_contract_rules;
- отвечай только по найденным фрагментам;
- не придумывай отсутствующие условия;
- если информации недостаточно, скажи об этом;
- указывай раздел, на котором основан ответ.
""",
)

result = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": (
                "Может ли исполнитель самостоятельно "
                "повысить стоимость услуг?"
            ),
        }
    ]
})

print(result["messages"][-1].content)