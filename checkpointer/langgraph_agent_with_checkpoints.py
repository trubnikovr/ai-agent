from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from ai.ai_client import ai_client


# Хранит checkpoints в памяти текущего процесса.
checkpointer = InMemorySaver()

agent = create_agent(
    model=ai_client.client,
    tools=[],
    system_prompt=(
        "You are a helpful assistant. "
        "Remember information from the current conversation. "
        "Answer in the user's language."
    ),
    checkpointer=checkpointer,
)

# Все вызовы с этим thread_id относятся к одному диалогу.
config = {
    "configurable": {
        "thread_id": "conversation-1",
    },
    "recursion_limit": 20,
}


if __name__ == "__main__":
    print("Agent with checkpoints")
    print("Напиши 'exit', чтобы завершить диалог.\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() == "exit":
            print("До встречи!")
            break

        if not user_input:
            continue

        result = agent.invoke(
            {
                # Передаём только новое сообщение.
                "messages": [HumanMessage(content=user_input)]
            },
            config=config,
        )

        print(f"\nAssistant: {result['messages'][-1].text}\n")