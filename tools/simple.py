from langchain_core.tools import tool

from ai.ai_client import ai_client
llm = ai_client.client



# 1. Создаём обычную Python-функцию и объявляем её инструментом
@tool
def calculate_penalty(
    amount: float,
    days_overdue: int,
    daily_rate: float,
) -> float:
    """
    Рассчитывает неустойку за просрочку.

    amount — сумма обязательства;
    days_overdue — количество дней просрочки;
    daily_rate — дневная ставка в процентах.
    """

    return amount * days_overdue * daily_rate / 100

# 3. Сообщаем модели о доступных инструментах
llm_with_tools = llm.bind_tools([
    calculate_penalty,
])

# 4. Отправляем вопрос
response = llm_with_tools.invoke(
    """
    Рассчитай неустойку:

    Сумма обязательства: 1 000 000 сум.
    Просрочка: 10 дней.
    Ставка: 0.1% в день.
    """
)

print("Текст ответа:")
print(response.content)


if response.tool_calls:
    tool_call = response.tool_calls[0]

    if tool_call["name"] == calculate_penalty.name:
        tool_result = calculate_penalty.invoke(
            tool_call["args"]
        )

        print("\nРезультат инструмента:")
        print(tool_result)