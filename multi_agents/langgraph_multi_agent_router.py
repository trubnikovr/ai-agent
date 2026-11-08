from typing import Literal

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from pydantic import BaseModel, Field

from ai.ai_client import ai_client


# ---------- Tools ----------


@tool
def search_travel_info(town: str) -> str:
    """Find tourist information about a town in Cornwall."""
    destinations = {
        "newquay": "A beach town popular for surfing.",
        "st ives": "A coastal town known for beaches and art galleries.",
    }

    return destinations.get(
        town.strip().casefold(),
        f"No travel information found for {town}.",
    )


@tool
def search_accommodation(town: str) -> list[dict]:
    """Find sample accommodation options in a town in Cornwall."""
    accommodations = {
        "newquay": [
            {"name": "Ocean View Hotel", "price_per_night": 120},
            {"name": "Surf House B&B", "price_per_night": 80},
        ],
        "st ives": [
            {"name": "Harbour Hotel", "price_per_night": 150},
            {"name": "Coastal B&B", "price_per_night": 95},
        ],
    }

    return accommodations.get(town.strip().casefold(), [])


# ---------- Specialized agents ----------


travel_agent = create_agent(
    model=ai_client.client,
    tools=[search_travel_info],
    system_prompt=(
        "You are a travel information agent. "
        "Use your tool to answer questions about destinations. "
        "Answer in the user's language."
    ),
)

booking_agent = create_agent(
    model=ai_client.client,
    tools=[search_accommodation],
    system_prompt=(
        "You are an accommodation agent. "
        "Use your tool to find accommodation options. "
        "The data is fictional: do not claim to make real bookings. "
        "Answer in the user's language."
    ),
)


# ---------- Router ----------


class Route(BaseModel):
    destination: Literal["travel", "booking"] = Field(
        description=(
            "Choose 'travel' for questions about destinations and "
            "'booking' for questions about accommodation."
        )
    )


router_model = ai_client.client.with_structured_output(Route)


def choose_agent(state: MessagesState) -> Literal["travel", "booking"]:
    """Choose a specialized agent based on the user's request."""
    user_message = state["messages"][-1]

    route = router_model.invoke(
        [
            HumanMessage(
                content=(
                    "Route this request to the appropriate agent.\n\n"
                    f"User request: {user_message.content}"
                )
            )
        ]
    )

    print(f"\n[Router] Selected agent: {route.destination}\n")
    return route.destination


# ---------- Graph nodes ----------


def run_travel_agent(state: MessagesState) -> dict:
    result = travel_agent.invoke({"messages": state["messages"]})

    # Добавляем в общий state финальный ответ agent.
    return {"messages": [result["messages"][-1]]}


def run_booking_agent(state: MessagesState) -> dict:
    result = booking_agent.invoke({"messages": state["messages"]})

    return {"messages": [result["messages"][-1]]}


# ---------- Multi-agent graph ----------


graph = StateGraph(MessagesState)

graph.add_node("travel", run_travel_agent)
graph.add_node("booking", run_booking_agent)

graph.add_conditional_edges(
    START,
    choose_agent,
    {
        "travel": "travel",
        "booking": "booking",
    },
)

graph.add_edge("travel", END)
graph.add_edge("booking", END)

multi_agent = graph.compile()


if __name__ == "__main__":
    question = input("You: ").strip()

    if question:
        result = multi_agent.invoke(
            {"messages": [HumanMessage(content=question)]}
        )

        print("\n--- Answer ---\n")
        result["messages"][-1].pretty_print()