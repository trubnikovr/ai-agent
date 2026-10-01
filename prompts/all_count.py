from langchain_core.messages import HumanMessage, SystemMessage

from ai.ai_client import ai_client


llm = ai_client.client


system_prompt = """
<Persona>
Ты преподаватель PostgreSQL.
</Persona>

<Instructions>
Объясняй понятия простым языком.
Используй небольшой практический пример.
</Instructions>

<OutputFormat>
Дай ответ не более чем в трёх абзацах.
</OutputFormat>
"""


user_prompt = """
<Task>
Объясни, что такое транзакция PostgreSQL.
</Task>
"""


messages = [
    SystemMessage(content=system_prompt),
    HumanMessage(content=user_prompt),
]


response = llm.invoke(messages)


print("Ответ:")
print(response.text)

print("\nИспользование токенов:")
print(response.usage_metadata)

print("\nMetadata ответа:")
print(response.response_metadata)