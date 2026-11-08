from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from ai.ai_client import ai_client


# Учебные данные вместо поиска в vector store.
DESTINATIONS = {
    "Newquay": "A popular beach town known for surfing.",
    "St Ives": "A coastal town with sandy beaches and art galleries.",
    "Falmouth": "A harbour town with beaches and maritime attractions.",
    "Perranporth": "A seaside town with a long sandy beach.",
}

# Mock-данные: это не реальный прогноз погоды.
WEATHER = {
    "Newquay": {"weather": "rainy", "temperature_c": 16},
    "St Ives": {"weather": "sunny", "temperature_c": 22},
    "Falmouth": {"weather": "sunny", "temperature_c": 21},
    "Perranporth": {"weather": "windy", "temperature_c": 18},
}


@tool
def search_travel_info(query: str) -> dict:
    """Find beach towns in Cornwall and return information about them.

    Use this tool to find candidate destinations before checking weather.
    """
    return {
        "query": query,
        "destinations": DESTINATIONS,
    }


@tool
def weather_forecast(town: str) -> dict:
    """Get a sample weather forecast for a given Cornwall town.

    The forecast is mock data and does not represent real weather.
    """
    for known_town, forecast in WEATHER.items():
        if known_town.casefold() == town.strip().casefold():
            return {
                "town": known_town,
                **forecast,
            }

    return {
        "error": f"No weather data found for '{town}'.",
        "available_towns": list(WEATHER),
    }


tools = [search_travel_info, weather_forecast]

# Используем модель из твоего AIClient.
model = ai_client.client
model_with_tools = model.bind_tools(tools)

system_message = SystemMessage(
    content=(
        "You are a travel assistant for Cornwall. "
        "Use search_travel_info to find destinations. "
        "Use weather_forecast when the user asks about weather. "
        "Base your recommendations on tool results. "
        "Mention that weather data is simulated, not live. "
        "Answer in the user's language."
    )
)


def call_model(state: MessagesState) -> dict:
    """Call the model with the current conversation."""
    response = model_with_tools.invoke(
        [system_message, *state["messages"]]
    )

    return {"messages": [response]}


def choose_next_step(state: MessagesState) -> Literal["tools", "end"]:
    """Decide whether to execute tools or finish."""
    last_message = state["messages"][-1]

    if last_message.tool_calls:
        return "tools"

    return "end"


# Создаём graph и добавляем nodes.
graph = StateGraph(MessagesState)

graph.add_node("agent", call_model)
graph.add_node("tools", ToolNode(tools))

# Первый node — agent.
graph.add_edge(START, "agent")

# После agent идём либо к tools, либо завершаем работу.
graph.add_conditional_edges(
    "agent",
    choose_next_step,
    {
        "tools": "tools",
        "end": END,
    },
)

# После выполнения tools возвращаем результаты модели.
graph.add_edge("tools", "agent")

agent = graph.compile()


if __name__ == "__main__":
    question = "Посоветуй два пляжных города в Корнуолле с хорошей погодой."

    result = agent.invoke(
        {"messages": [HumanMessage(content=question)]},
        config={"recursion_limit": 20},
    )

    print("\n--- Agent execution ---\n")

    for message in result["messages"]:
        message.pretty_print()