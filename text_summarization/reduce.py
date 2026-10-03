reduce_prompt = ChatPromptTemplate.from_messages([
    ("system", """
<Instructions>
Объедини краткие пересказы частей книги в одно общее резюме.
Сохрани главные события и факты.
Убери повторы.
Не добавляй того, чего нет в переданных пересказах.
</Instructions>
"""),
    ("human", """
<ChunkSummaries>
{summaries}
</ChunkSummaries>
"""),
])

summaries_text = "\n\n".join(chunk_summaries)

messages = reduce_prompt.invoke({
    "summaries": summaries_text,
})

response = ai_client.client.invoke(messages)

final_summary = response.content
print(final_summary)