import asyncio
import re

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient

from ai.ai_client import ai_client


# Локальные туристические заметки для демонстрации.
# В этом примере поиск простой, по совпадающим словам.
TRAVEL_NOTES = [
    (
        "Ташкент: можно посетить комплекс Хазрати Имам, "
        "базар Чорсу и музей прикладного искусства."
    ),
    (
        "Самарканд: главные достопримечательности — Регистан, "
        "мавзолей Гур-Эмир и комплекс Шахи-Зинда."
    ),
    (
        "Перед прогулкой по старому городу возьмите воду "
        "и проверьте прогноз погоды."
    ),
]


@tool
def search_travel_info(query: str) -> str:
    """Ищет сведения о городах и достопримечательностях Узбекистана."""

    query_words = set(re.findall(r"\w+", query.lower()))

    scored_notes = []

    for note in TRAVEL_NOTES:
        note_words = set(re.findall(r"\w+", note.lower()))
        score = len(query_words & note_words)

        if score > 0:
            scored_notes.append((score, note))

    scored_notes.sort(reverse=True)

    if not scored_notes:
        return "В локальных туристических заметках ничего не найдено."

    return "\n".join(note for _, note in scored_notes[:3])


async def chat_loop(agent) -> None:
    messages = []

    while True:
        user_text = await asyncio.to_thread(input, "\nТы: ")

        if user_text.strip().lower() in {"выход", "exit", "quit"}:
            print("Чат завершён.")
            break

        messages.append({
            "role": "user",
            "content": user_text,
        })

        result = await agent.ainvoke({"messages": messages})

        # История включает сообщения пользователя, агента,
        # вызовы инструментов и результаты их выполнения.
        messages = result["messages"]

        print("\nАгент:", messages[-1].content)


async def main() -> None:
    mcp_client = MultiServerMCPClient(
        {
            "weather": {
                "transport": "http",
                "url": "http://127.0.0.1:8020/mcp",
            }
        }
    )

    # Загружаем инструмент get_weather_conditions с MCP-сервера.
    weather_tools = await mcp_client.get_tools()

    # Объединяем локальные инструменты с MCP-инструментами.
    tools = [
        search_travel_info,
        *weather_tools,
    ]

    print("Инструменты агента:")
    for tool_item in tools:
        print(f"- {tool_item.name}: {tool_item.description}")

    agent = create_agent(
        model=ai_client.client,
        tools=tools,
        system_prompt=(
            "Ты помощник по путешествиям в Узбекистане. "
            "Для вопросов о достопримечательностях используй "
            "search_travel_info. Для вопросов о погоде используй "
            "get_weather_conditions. Если вопрос требует и того и другого, "
            "можешь использовать оба инструмента. Отвечай на русском."
        ),
    )

    await chat_loop(agent)


if __name__ == "__main__":
    asyncio.run(main())

