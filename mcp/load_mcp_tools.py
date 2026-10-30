import asyncio

from langchain_mcp_adapters.client import MultiServerMCPClient


async def main() -> None:
    mcp_client = MultiServerMCPClient(
        {
            "weather": {
                "transport": "http",
                "url": "http://127.0.0.1:8020/mcp",
            }
        }
    )

    # Получаем MCP-инструменты и преобразуем их в инструменты LangChain.
    tools = await mcp_client.get_tools()

    print("Инструменты, доступные агенту:")
    for tool in tools:
        print(f"- {tool.name}: {tool.description}")

    # Пока вызываем инструмент напрямую, без LLM.
    weather_tool = next(
        tool for tool in tools
        if tool.name == "get_weather_conditions"
    )

    result = await weather_tool.ainvoke({"location": "Ташкент"})
    print("\nРезультат вызова:")
    print(result)


if __name__ == "__main__":
    asyncio.run(main())