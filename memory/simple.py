from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from ai.ai_client import ai_client


llm = ai_client.client

memory = InMemorySaver()

agent = create_agent(
    model=llm,
    tools=[],
    checkpointer=memory,
    system_prompt="Ты полезный AI-ассистент.",
)


config = {
    "configurable": {
        "thread_id": "user-123",
    }
}


# Первый запрос
result_1 = agent.invoke(
    {
        "messages": [
            HumanMessage(
                content="Меня зовут Рустам. Я backend-разработчик."
            )
        ]
    },
    config=config,
)

print(result_1["messages"][-1].content)


# Второй запрос
result_2 = agent.invoke(
    {
        "messages": [
            HumanMessage(
                content="Как меня зовут и чем я занимаюсь?"
            )
        ]
    },
    config=config,
)

print(result_2["messages"][-1].content)