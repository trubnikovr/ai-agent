from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain.agents import create_agent
from ai.ai_client import ai_client
from config import configure_langchain

configure_langchain()

DESTINATIONS = {
  "Newquay": "A popular beach town known for surfing.",
  "St Ives": "A coastal town with sandy beaches and art galleries.",
  "Falmouth": "A harbour town with beaches and maritime attractions.",
  "Perranporth": "A seaside town with a long sandy beach.",
}

# Mock-данные, не реальный прогноз.
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
  """Get a mock weather forecast for a given Cornwall town.

  The forecast is simulated and does not represent real weather.
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

system_prompt = (
  "You are a travel assistant for Cornwall. "
  "Use search_travel_info to find destinations. "
  "Use weather_forecast when the user asks about weather. "
  "Base your recommendations on tool results. "
  "Mention that weather data is simulated, not live. "
  "Answer in the user's language."
)

# Prebuilt agent сам связывает model с tools и собирает graph.
agent = create_agent(
    model=ai_client.client,
    tools=tools,
    system_prompt=system_prompt,
)

#print(agent.get_graph().draw_mermaid())


if __name__ == "__main__":
  messages = []

  print("Cornwall Travel Assistant")
  print("Напиши 'exit', чтобы завершить диалог.\n")

  while True:
      user_input = input("You: ").strip()

      if user_input.lower() == "exit":
          print("До встречи!")
          break

      if not user_input:
          continue

      messages.append(HumanMessage(content=user_input))
      previous_count = len(messages)

      result = agent.invoke(
          {"messages": messages},
          config={"recursion_limit": 20},
      )

      messages = result["messages"]

      print("\n--- Agent execution ---\n")

      for message in messages[previous_count:]:
          message.pretty_print()

      print()