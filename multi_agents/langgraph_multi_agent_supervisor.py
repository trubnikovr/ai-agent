from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from ai.ai_client import ai_client


# ---------- Tools для специализированных agents ----------


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


# ---------- Специализированные agents ----------


travel_agent = create_agent(
  model=ai_client.client,
  tools=[search_travel_info],
  system_prompt=(
      "You are a travel information agent. "
      "Use your tool to answer questions about destinations. "
      "Answer in the user's language."
  ),
)

accommodation_agent = create_agent(
  model=ai_client.client,
  tools=[search_accommodation],
  system_prompt=(
      "You are an accommodation agent. "
      "Use your tool to find accommodation options. "
      "The data is fictional; do not claim to make real bookings. "
      "Answer in the user's language."
  ),
)


# ---------- Tools для вызова agents ----------


@tool
def ask_travel_agent(question: str) -> str:
  """Ask the travel agent about a destination in Cornwall."""
  result = travel_agent.invoke(
      {"messages": [HumanMessage(content=question)]},
      config={"recursion_limit": 15},
  )

  return result["messages"][-1].text


@tool
def ask_accommodation_agent(question: str) -> str:
  """Ask the accommodation agent to find sample places to stay."""
  result = accommodation_agent.invoke(
      {"messages": [HumanMessage(content=question)]},
      config={"recursion_limit": 15},
  )

  return result["messages"][-1].text


# ---------- Supervisor ----------


supervisor = create_agent(
  model=ai_client.client,
  tools=[ask_travel_agent, ask_accommodation_agent],
  system_prompt=(
      "You coordinate two specialized agents. "
      "Use the travel agent for information about destinations. "
      "Use the accommodation agent for places to stay. "
      "If the user needs both, consult both agents. "
      "Combine their answers into one response. "
      "Accommodation data is fictional; do not claim to make real bookings. "
      "Answer in the user's language."
  ),
)


if __name__ == "__main__":
  question = input("You: ").strip()

  if question:
      result = supervisor.invoke(
          {"messages": [HumanMessage(content=question)]},
          config={"recursion_limit": 20},
      )

      print("\n--- Supervisor execution ---\n")

      for message in result["messages"]:
          message.pretty_print()