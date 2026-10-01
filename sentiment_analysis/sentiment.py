from typing import Literal

from langchain_core.prompts import HumanMessagePromptTemplate, SystemMessagePromptTemplate, ChatPromptTemplate
from pydantic import BaseModel, Field

from ai.ai_client import ai_client

llm = ai_client

class SentimentResult(BaseModel):
    sentiment: Literal["positive", "neutral", "negative"] = Field(
        description="Тональность текста"
    )

structured_llm = ai_client.client.with_structured_output(
    SentimentResult
)
system_message = SystemMessagePromptTemplate.from_template("""
<Persona>
Ты анализируешь тональность коротких текстов.
</Persona>

<Instructions>
Определи тональность текста. Выбери только одну категорию:
positive, neutral или negative.

Оценивай отношение, выраженное в тексте, а не отдельные слова.
</Instructions>

<OutputFormat>
Верни только название категории.
</OutputFormat>
""")

human_message = HumanMessagePromptTemplate.from_template("""
<Input>
{text}
</Input>
""")

prompt = ChatPromptTemplate.from_messages([
    system_message,
    human_message,
])

prompt_value = prompt.invoke({
    "text": "Apple сообщила о рекордной квартальной прибыли."
})

result = structured_llm.invoke(prompt_value)

print(result)
# SentimentResult(sentiment='positive')

print(result.sentiment)
# positive