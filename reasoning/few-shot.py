from langchain_core.prompts import PromptTemplate
from langchain_core.prompts.few_shot import FewShotPromptTemplate

from ai.ai_client import ai_client

examples = [
    {
        "number": 6,
        "reasoning": "не делится ни на 5, ни на 7",
        "result": "None",
    },
    {
        "number": 15,
        "reasoning": "делится на 5, но не на 7",
        "result": "Abra",
    },
    {
        "number": 21,
        "reasoning": "делится на 7, но не на 5",
        "result": "Kadabra",
    },
    {
        "number": 70,
        "reasoning": "делится и на 5, и на 7",
        "result": "Abra Kadabra",
    },
]

# Шаблон одного примера
example_prompt = PromptTemplate.from_template(
    "{number} // {reasoning} // {result}"
)

# Шаблон всего запроса
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    suffix="""
Классифицируй числа:
{numbers}
""",
    input_variables=["numbers"],
)

# Подставляем числа и собираем готовый промпт
prompt_text = few_shot_prompt.format(numbers="3, 5, 7, 35")

# Отправляем промпт нашей модели
response = ai_client.client.invoke(prompt_text)

print(response.content)