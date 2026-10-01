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


@tool
def calculate_vat(
    amount: float,
    vat_rate: float = 12,
) -> float:
    """
    Рассчитывает сумму НДС.

    amount — сумма без НДС;
    vat_rate — ставка НДС в процентах.
    """

    return amount * vat_rate / 100



agent = create_agent(
    model=llm,
    tools=[calculate_penalty, calculate_vat],
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
      Выполни два расчёта:

            1. Рассчитай неустойку с суммы 1 000 000 сум:
            просрочка — 10 дней;
            ставка — 0.1% в день.
            
            2. Рассчитай НДС 12% от суммы 1 000 000 сум.
        """
        )
    ]
})

for message in result["messages"]:
    if getattr(message, "tool_calls", None):
        for tool_call in message.tool_calls:
            print("Инструмент:", tool_call["name"])
            print("Аргументы:", tool_call["args"])