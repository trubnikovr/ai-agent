from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    MessagesPlaceholder,
)


system_message = SystemMessagePromptTemplate.from_template("""
<Persona>
Ты преподаватель {technology}.
</Persona>

<Instructions>
Объясняй понятия простым языком.
Используй небольшой пример.
</Instructions>

<OutputFormat>
Не более трёх абзацев.
</OutputFormat>
""")


human_message = HumanMessagePromptTemplate.from_template("""
<Task>
Объясни понятие: {topic}.
</Task>
""")


prompt = ChatPromptTemplate.from_messages([
    system_message,
    human_message,
])