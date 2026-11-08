from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from ai.ai_client import ai_client


# ---------- Agent ----------
@tool
def weather_forecast(town: str) -> dict:
  """Get a mock weather forecast for a town."""
  weather = {
      "newquay": {"weather": "rainy", "temperature_c": 16},
      "st ives": {"weather": "sunny", "temperature_c": 22},
  }

  forecast = weather.get(town.strip().casefold())

  if forecast is None:
      return {
          "error": f"No weather data found for '{town}'.",
      }

  return {
      "town": town,
      **forecast,
      "source": "mock data",
  }


agent = create_agent(
  model=ai_client.client,
  tools=[weather_forecast],
  system_prompt=(
      "You are a weather assistant. "
      "Use weather_forecast for questions about weather. "
      "The weather data is simulated, not live. "
      "Answer in the user's language."
  ),
)


# ---------- Eval dataset ----------


examples = [
  {
      "name": "weather in Newquay",
      "input": "Какая погода в Newquay?",
      "expected_tool": "weather_forecast",
      "expected_town": "Newquay",
  },
  {
      "name": "weather in St Ives",
      "input": "Какая погода в St Ives?",
      "expected_tool": "weather_forecast",
      "expected_town": "St Ives",
  },
  {
      "name": "umbrella in Newquay",
      "input": "Нужен ли мне зонт в Newquay?",
      "expected_tool": "weather_forecast",
      "expected_town": "Newquay",
  },
  {
      "name": "greeting",
      "input": "Привет!",
      "expected_tool": None,
      "expected_town": None,
  },
]


# ---------- Evaluator ----------


def get_tool_calls(messages: list) -> list[dict]:
  """Collect all tool calls from agent messages."""
  calls = []

  for message in messages:
      if isinstance(message, AIMessage):
          calls.extend(message.tool_calls)

  return calls


def evaluate(example: dict) -> dict:
  """Run one example and check the agent's tool calls."""
  result = agent.invoke(
      {"messages": [HumanMessage(content=example["input"])]},
      config={"recursion_limit": 20},
  )

  tool_calls = get_tool_calls(result["messages"])

  expected_tool = example["expected_tool"]
  expected_town = example["expected_town"]

  if expected_tool is None:
      # Для приветствия ожидаем, что agent не вызовет tools.
      tool_correct = len(tool_calls) == 0
      town_correct = True
  else:
      matching_calls = [
          call for call in tool_calls
          if call["name"] == expected_tool
      ]

      tool_correct = len(matching_calls) > 0

      town_correct = any(
          str(call["args"].get("town", "")).strip().casefold()
          == expected_town.casefold()
          for call in matching_calls
      )

  return {
      "name": example["name"],
      "input": example["input"],
      "tool_correct": tool_correct,
      "town_correct": town_correct,
      "passed": tool_correct and town_correct,
      "tool_calls": tool_calls,
      "answer": result["messages"][-1].text,
  }


# ---------- Run eval ----------


if __name__ == "__main__":
  results = []

  for example in examples:
      print(f"\nRunning: {example['name']}")

      result = evaluate(example)
      results.append(result)

      status = "PASS" if result["passed"] else "FAIL"

      print(f"  Status: {status}")
      print(f"  Tool correct: {result['tool_correct']}")
      print(f"  Town correct: {result['town_correct']}")
      print(f"  Tool calls: {result['tool_calls']}")
      print(f"  Answer: {result['answer']}")

  passed = sum(result["passed"] for result in results)
  total = len(results)

  print("\n--- Eval summary ---")
  print(f"Passed: {passed}/{total}")
  print(f"Success rate: {passed / total:.0%}")
