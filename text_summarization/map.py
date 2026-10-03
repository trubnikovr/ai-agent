from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import TokenTextSplitter

from ai.ai_client import ai_client

splitter = TokenTextSplitter(
    chunk_size=3000,
    chunk_overlap=100,
)
llm = ai_client

file_path = Path(__file__).parent / "Moby-Dick.txt"

with file_path.open("r", encoding="utf-8") as file:
    document_text = file.read()

chunks = splitter.split_text(document_text)

chunk_prompt = ChatPromptTemplate.from_messages([
    ("system", """
<Instructions>
Кратко перескажи переданный фрагмент.
Сохрани главные события и факты.
Не добавляй сведений, которых нет в тексте.
</Instructions>
"""),
    ("human", """
<Chunk>
{text}
</Chunk>
"""),
])

chunk_summaries = []
for index, chunk in enumerate(chunks, start=1):
    # Подставляем текущий фрагмент в шаблон
    messages = chunk_prompt.invoke({"text": chunk})

    # Отправляем готовые сообщения модели
    response = ai_client.client.invoke(messages)

    # Сохраняем отдельное резюме
    chunk_summaries.append(response.content)
    print(f"Готово: {index}/{len(chunks)}")