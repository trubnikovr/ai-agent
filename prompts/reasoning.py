prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("""
<Persona>
Ты помощник по математике.
</Persona>

<Instructions>
Проверь каждое число по заданному условию.
Не добавляй числа, которых нет в диапазоне.
</Instructions>

<OutputFormat>
Перечисли подходящие числа и кратко объясни ответ.
</OutputFormat>
"""),
    HumanMessagePromptTemplate.from_template("""
<Task>
Какие числа от 1 до 10 являются квадратами целых чисел?
</Task>
"""),
])

messages = prompt.invoke({})
response = ai_client.client.invoke(messages)

print(response.content)