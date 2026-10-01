from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("""
<Persona>
Ты помощник, который отвечает на вопросы по документам.
</Persona>

<Instructions>
Отвечай только на основании текста в блоке <SourceText>.
Не используй внешние знания.
Если ответа в тексте нет, скажи: «В документе это не указано».
</Instructions>

<OutputFormat>
Ответь кратко: одно-два предложения.
</OutputFormat>
"""),
    HumanMessagePromptTemplate.from_template("""
<SourceText>
{text}
</SourceText>

<Question>
{question}
</Question>
"""),
])

source_text = """
Компания обязуется доставить оборудование до 15 октября.
Оплата производится в течение десяти рабочих дней после доставки.
"""

messages = prompt.invoke({
    "text": source_text,
    "question": "В какой срок нужно оплатить оборудование?",
})

response = ai_client.client.invoke(messages)
print(response.content)