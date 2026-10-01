import json
from typing import Literal

from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from pydantic import BaseModel, Field

from ai.ai_client import ai_client

Sentiment = Literal["positive", "neutral", "negative"]


class StockSentiment(BaseModel):
    company: str = Field(
        description="Название компании из входного списка"
    )

    sentiment: Sentiment = Field(
        description="Тональность новости для этой компании"
    )


class SentimentBatch(BaseModel):
    results: list[StockSentiment] = Field(
        description="Тональность каждой новости из входного списка"
    )


system_message = SystemMessagePromptTemplate.from_template("""
<Persona>
Ты анализируешь финансовые новости.
</Persona>

<Instructions>
Для каждой новости определи тональность для указанной компании:

- positive — новость скорее благоприятна для компании;
- neutral — новость не выражает явно положительную или отрицательную оценку;
- negative — новость скорее неблагоприятна для компании.

Не оценивай новость с точки зрения эмоций автора.
Оценивай вероятное влияние описанного события на компанию.
Обработай каждую новость ровно один раз.
Не добавляй компании, которых нет во входных данных.
</Instructions>

<OutputFormat>
Верни результат в заданной структуре.
</OutputFormat>
""")

human_message = HumanMessagePromptTemplate.from_template("""
<NewsItems>
{news_json}
</NewsItems>
""")

prompt = ChatPromptTemplate.from_messages([
    system_message,
    human_message,
])

news = [
    {
        "company": "Apple",
        "text": "Запуск нового iPhone оказался успешным.",
    },
    {
        "company": "Nvidia",
        "text": "Спрос на процессоры для систем искусственного интеллекта вырос.",
    },
    {
        "company": "GX Oil",
        "text": "Спрос на нефть снизился на фоне роста использования возобновляемой энергии.",
    },
]

prompt_value = prompt.invoke({
    "news_json": json.dumps(news, ensure_ascii=False, indent=2)
})

structured_llm = ai_client.client.with_structured_output(
    SentimentBatch
)

result = structured_llm.invoke(prompt_value)

print("| Компания | Тональность |")
print("|---|---|")

for item in result.results:
    print(f"| {item.company} | {item.sentiment} |")