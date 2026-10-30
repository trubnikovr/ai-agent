import asyncio

from fastmcp import Client


async def main() -> None:
    async with Client("http://127.0.0.1:8020/mcp") as client:
        # Узнаём, какие инструменты предоставляет сервер.
        tools = await client.list_tools()

        print("Инструменты сервера:")
        for tool in tools:
            print(f"- {tool.name}: {tool.description}")
            print("  Схема аргументов:", tool.input_schema)

        # Вызываем один из найденных инструментов.
        result = await client.call_tool(
            "get_weather_conditions",
            {"location": "Ташкент"},
        )

        print("\nРезультат:")
        print(result.data)


if __name__ == "__main__":
    asyncio.run(main())