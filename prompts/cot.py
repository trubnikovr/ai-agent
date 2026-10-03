from langchain_core.prompts import ChatPromptTemplate

from ai.ai_client import ai_client

prompt = ChatPromptTemplate.from_messages([
    ("system", """
<Term>
Последовательность Strange, если в ней есть как минимум два нечётных числа,
а сумма всех нечётных чисел делится на 3.
</Term>

<Instructions>
Проверь числа по порядку:
1. Выпиши нечётные числа.
2. Проверь, что их как минимум два.
3. Посчитай их сумму и проверь делимость на 3.
4. Верни только итог и краткое объяснение.
</Instructions>

<Examples>
Последовательность: 5, 6, 7, 8, 20
Нечётные: 5, 7. Их два, сумма 12 делится на 3.
Результат: Strange.

Последовательность: 1, 5, 6, 7, 8, 20
Нечётные: 1, 5, 7. Их сумма 13 не делится на 3.
Результат: Not Strange.
</Examples>
"""),
    ("human", """
Проверь последовательность:
{numbers}
"""),
])

messages = prompt.invoke({"numbers": "3, 4, 5, 7, 10, 18, 22, 24"})
response = ai_client.client.invoke(messages)

print(response.content)