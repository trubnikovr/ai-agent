from typing import Literal

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from pydantic import BaseModel, Field

from ai.ai_client import ai_client


# ---------- Tool ----------


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


# ---------- Travel agent ----------


travel_agent = create_agent(
  model=ai_client.client,
  tools=[search_travel_info],
  system_prompt=(
      "You are a travel assistant. "
      "Use the available tool for destination information. "
      "Answer in the user's language."
  ),
)


# ---------- Guardrail ----------


class TopicCheck(BaseModel):
  is_travel_related: bool = Field(
      description="Whether the user's request is related to travel."
  )


guardrail_model = ai_client.client.with_structured_output(TopicCheck)


def check_topic(state: MessagesState) -> Literal["travel", "reject"]:
  """Check whether the user's request is related to travel."""
  user_message = state["messages"][-1]

  result = guardrail_model.invoke(
      [
          HumanMessage(
              content=(
                  "Is this request related to travel, tourism, "
                  "destinations, transport, or accommodation?\n\n"
                  f"Request: {user_message.content}"
              )
          )
      ]
  )

  print(f"\n[Guardrail] Travel related: {result.is_travel_related}")

  if result.is_travel_related:
      return "travel"

  return "reject"


# ---------- Graph nodes ----------


def run_travel_agent(state: MessagesState) -> dict:
  result = travel_agent.invoke(
      {"messages": state["messages"]},
      config={"recursion_limit": 20},
  )

  return {"messages": [result["messages"][-1]]}


def reject_request(state: MessagesState) -> dict:
  return {
      "messages": [
          AIMessage(
              content=(
                  "Я помогаю с вопросами о путешествиях. "
                  "Спроси меня о поездках или туристических местах."
              )
          )
      ]
  }


# ---------- Graph ----------


graph = StateGraph(MessagesState)

graph.add_node("travel", run_travel_agent)
graph.add_node("reject", reject_request)

graph.add_conditional_edges(
  START,
  check_topic,
  {
      "travel": "travel",
      "reject": "reject",
  },
)

graph.add_edge("travel", END)
graph.add_edge("reject", END)

guarded_agent = graph.compile()


if __name__ == "__main__":
  question = input("You: ").strip()

  if question:
      result = guarded_agent.invoke(
          {"messages": [HumanMessage(content=question)]}
      )

      print("\n--- Answer ---\n")
      result["messages"][-1].pretty_print()