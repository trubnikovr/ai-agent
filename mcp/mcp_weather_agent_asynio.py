import asyncio

from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient

from ai.ai_client import ai_client


async def chat_loop(agent_with_mcp) -> None:
    messages = []

    while True:
        user_text = await asyncio.to_thread(input, "Ты: ")

        if user_text.strip().lower() in {"выход", "exit", "quit"}:
            break

        messages.append({
            "role": "user",
            "content": user_text,
        })

        result = await agent_with_mcp.ainvoke({"messages": messages})

        # Сохраняем всю историю: в ней могут быть вызов инструмента
        # и его результат, кроме реплик пользователя и агента.
        messages = result["messages"]

        print("Агент:", messages[-1].content)


async def main() -> None:
    mcp_client = MultiServerMCPClient(
        {
            "weather": {
                "transport": "http",
                "url": "http://127.0.0.1:8020/mcp",
            }
        }
    )

    remote_tools = await mcp_client.get_tools()

    agent_with_mcp = create_agent(
        model=ai_client.client,
        tools=remote_tools,
        system_prompt=(
            "Ты помощник по погоде. "
            "Для вопросов о погоде используй get_weather_conditions. "
            "Отвечай на русском."
        ),
    )

    await chat_loop(agent_with_mcp)


if __name__ == "__main__":
    asyncio.run(main())