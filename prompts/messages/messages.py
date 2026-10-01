from langchain_core.messages import (
    # Базовый класс
    BaseMessage,

    # Основные сообщения
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,

    # Дополнительные сообщения
    ChatMessage,
    FunctionMessage,
    RemoveMessage,

    # Потоковые части сообщений
    SystemMessageChunk,
    HumanMessageChunk,
    AIMessageChunk,
    ToolMessageChunk,
    ChatMessageChunk,
    FunctionMessageChunk,
)