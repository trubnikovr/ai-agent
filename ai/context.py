from langchain_core.messages import BaseMessage, trim_messages
from langchain_google_genai import ChatGoogleGenerativeAI


def trim_context(
    llm: ChatGoogleGenerativeAI,
    messages: list[BaseMessage],
    max_tokens: int,
) -> list[BaseMessage]:
    return trim_messages(
        messages,
        strategy="last",
        token_counter=llm,
        max_tokens=max_tokens,
        start_on="human",
        include_system=True,
        allow_partial=False,
    )