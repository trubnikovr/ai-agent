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

tools = [
    calculate_penalty,
]

tools_by_name = {
    tool.name: tool
    for tool in tools
}

llm_with_tools = llm.bind_tools(tools)


messages = [
    HumanMessage(
        content="""
Рассчитай неустойку.

Сумма обязательства: 1 000 000 сум.
Просрочка: 10 дней.
Ставка: 0.1% в день.
"""
    )
]

# --------------------------------------------------
# 2. Первый вызов Gemini
# --------------------------------------------------

ai_message = llm_with_tools.invoke(messages)

messages.append(ai_message)


# --------------------------------------------------
# 3. Выполняем запрошенные инструменты
# --------------------------------------------------

for tool_call in ai_message.tool_calls:
    tool_name = tool_call["name"]

    selected_tool = tools_by_name[tool_name]

    tool_message = selected_tool.invoke(tool_call)

    messages.append(tool_message)

    print("\nРезультат инструмента:")
    print(tool_message.content)


# --------------------------------------------------
# 4. Снова вызываем Gemini
# --------------------------------------------------
# Повторно передаём описание инструментов. Для Gemini tool response должен
# обрабатываться в том же tool-enabled контексте, что и исходный запрос.
final_response = llm_with_tools.invoke(messages)
print("Final content:", final_response.content)
print("Final tool calls:", final_response.tool_calls)
