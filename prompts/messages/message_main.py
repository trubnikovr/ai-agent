from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage

messages: list[BaseMessage] = [
    SystemMessage(content="Ты преподаватель"),
    HumanMessage(content="Что такое транзакция?"),
]
for message in messages:
    message.content               # содержимое
    message.id                    # идентификатор
    message.name                  # необязательное имя
    message.additional_kwargs     # дополнительные данные провайдера
    message.response_metadata     # технические метаданные
    message.type                  # тип сообщения


tool_message = ToolMessage(
    content="10000",
    tool_call_id="call_123",
)

tool_message = ToolMessage(
    content="Найдено 5 документов",
    artifact={
        "documents": [...],
        "execution_time": 0.42,
    },
    tool_call_id="call_123",
)