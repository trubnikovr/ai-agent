from fastmcp import FastMCP
from pydantic import BaseModel, Field


mcp = FastMCP("Mock Weather MCP Server")


class WeatherResult(BaseModel):
    city: str = Field(description="Название города")
    temperature_c: float = Field(description="Температура в градусах Цельсия")
    condition: str = Field(description="Описание погоды")
    has_precipitation: bool = Field(description="Идут ли осадки")


MOCK_WEATHER = {
    "ташкент": WeatherResult(
        city="Ташкент",
        temperature_c=22.0,
        condition="Ясно",
        has_precipitation=False,
    ),
    "самарканд": WeatherResult(
        city="Самарканд",
        temperature_c=19.0,
        condition="Облачно",
        has_precipitation=False,
    ),
    "лондон": WeatherResult(
        city="Лондон",
        temperature_c=14.0,
        condition="Небольшой дождь",
        has_precipitation=True,
    ),
}


@mcp.tool
def get_weather_conditions(location: str) -> WeatherResult:
    """Получить демонстрационные погодные данные для города."""

    city_key = location.strip().lower()

    weather = MOCK_WEATHER.get(city_key)

    if weather is None:
        raise ValueError(
            f"Для города «{location}» нет mock-данных. "
            f"Доступны: {', '.join(item.city for item in MOCK_WEATHER.values())}"
        )

    return weather


if __name__ == "__main__":
    mcp.run(
        transport="http",
        host="127.0.0.1",
        port=8020,
    )