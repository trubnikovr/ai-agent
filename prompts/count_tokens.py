from langchain_core.messages import SystemMessage, HumanMessage

from ai.ai_client import ai_client
from ai.context import trim_context

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
messages = trim_context(messages=messages, llm=llm, max_tokens=200)
print(llm.get_num_tokens_from_messages(messages=messages))
exit
# count_result = ai_client.models.count_tokens(
#     model=ai_client,
#     contents=[
#         types.Content(
#             role="user",
#             parts=[
#                 types.Part.from_text(
#                     text=user_prompt,
#                 )
#             ],
#         )
#     ],
#     config=types.CountTokensConfig(
#         system_instruction=system_prompt,
#     ),
# )



exit
# count_result = ai_client.models.count_tokens(
#     model=ai_client,
#     contents=[
#         types.Content(
#             role="user",
#             parts=[
#                 types.Part.from_text(
#                     text=user_prompt,
#                 )
#             ],
#         )
#     ],
#     config=types.CountTokensConfig(
#         system_instruction=system_prompt,
#     ),
# )
