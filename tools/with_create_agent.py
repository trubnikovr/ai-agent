from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

from ai.ai_client import ai_client
llm = ai_client.client


@tool
def calculate_penalty(
    amount: float,
    days_overdue: int,
    daily_rate: float,
) -> float:
    """
    Рассчитывает неустойку.

    amount — сумма обязательства;
    days_overdue — количество дней просрочки;
    daily_rate — дневная процентная ставка.
    """

    return amount * days_overdue * daily_rate / 100



agent = create_agent(
    model=llm,
    tools=[calculate_penalty],
    system_prompt="""
Ты финансовый помощник.

Если для ответа нужен точный расчёт:
1. Используй подходящий инструмент.
2. Не рассчитывай результат самостоятельно.
3. После получения результата инструмента дай понятный ответ пользователю.
""",
)


result = agent.invoke({
    "messages": [
        HumanMessage(
            content="""
        Рассчитай неустойку.

        Сумма обязательства: 1 000 000 сум.
        Просрочка: 10 дней.
        Ставка: 0.1% в день.
        """
        )
    ]
})

for message in result["messages"]:
    message.pretty_print()
