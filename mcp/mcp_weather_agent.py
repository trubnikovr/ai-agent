import asyncio

from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient

from ai.ai_client import ai_client


mcp_client = MultiServerMCPClient(
    {
        "weather": {
            "transport": "http",
            "url": "http://127.0.0.1:8020/mcp",
        }
    }
)


async def main() -> None:
    remote_tools = await mcp_client.get_tools()

    agent = create_agent(
        model=ai_client.client,
        tools=remote_tools,
        system_prompt=(
            "Ты помощник по погоде. "
            "Для вопросов о погоде используй инструмент "
            "get_weather_conditions. Отвечай на русском."
        ),
    )

    result = await agent.ainvoke({
        "messages": [
            {"role": "user", "content": "Какая погода в Ташкенте?"}
        ]
    })

    print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())