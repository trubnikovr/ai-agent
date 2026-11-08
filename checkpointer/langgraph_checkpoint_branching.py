from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from ai.ai_client import ai_client


checkpointer = InMemorySaver()

agent = create_agent(
  model=ai_client.client,
  tools=[],
  system_prompt=(
      "You are a helpful travel assistant. "
      "Remember information from the conversation. "
      "Answer in the user's language."
  ),
  checkpointer=checkpointer,
)




def ask(question: str, run_config: dict) -> None:
  print(f"\nYou: {question}")

  result = agent.invoke(
      {"messages": [HumanMessage(content=question)]},
      config=run_config,
  )

  print(f"Assistant: {result['messages'][-1].text}")


config = {
  "configurable": {
      "thread_id": "branching-example",
  }
}

if __name__ == "__main__":
  # 1. Создаём общую отправную точку.
  ask("Меня зовут Рустам.", config)

  # 2. Получаем snapshot текущего state.
  snapshot = agent.get_state(config)

  # В snapshot.config содержится ссылка на конкретный checkpoint.
  checkpoint_config = snapshot.config

  print("\n--- Сохранили checkpoint ---")
  print("Checkpoint ID:", checkpoint_config["configurable"]["checkpoint_id"])

  # 3. Продолжаем первую ветку.
  print("\n--- Ветка 1 ---")

  ask("Я планирую поездку в Newquay.", config)
  ask("Куда я планирую поездку?", config)

  # 4. Возвращаемся к сохранённому checkpoint и создаём вторую ветку.
  print("\n--- Ветка 2: продолжение с сохранённого checkpoint ---")

  ask("Я планирую поездку в St Ives.", checkpoint_config)