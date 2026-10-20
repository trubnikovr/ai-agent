from dotenv import dotenv_values

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel


class AIClient:
    def __init__(self, api_key: str, model: str, temperature: float = 0):
        self._client = ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            google_api_key=api_key,
        )
        
    @property
    def client(self) -> ChatGoogleGenerativeAI:
        return self._client

    def ask(
        self,
        prompt: str,
        schema: type[BaseModel] | None = None,
    ):
        """
        Обычный запрос без ChatPromptTemplate.
        """

        if schema is None:
            response = self._client.invoke(prompt)
            return response.content

        structured_client = self._client.with_structured_output(schema)

        return structured_client.invoke(prompt)

    def ask_with_template(
        self,
        prompt: ChatPromptTemplate,
        schemas: list[type[BaseModel]] | None = None,
        **variables,
    ):
        """
        Запрос через ChatPromptTemplate.
        """

        if not schemas:
            chain = prompt | self._client
            response = chain.invoke(variables)

            return response.content

        chains = {
            schema.__name__: (
                prompt
                | self._client.with_structured_output(schema)
            )
            for schema in schemas
        }

        parallel_chain = RunnableParallel(chains)

        return parallel_chain.invoke(variables)


def create_ai_client(
    api_key: str | None = None,
    model: str = "gemini-3.5-flash-lite",
    temperature: float = 0,
) -> AIClient:
    if api_key is None:
        config = dotenv_values(".env")
        api_key = config.get("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("GEMINI_API_KEY не найден в .env")

    return AIClient(
        api_key=api_key,
        model=model,
        temperature=temperature,
    )


ai_client = create_ai_client()
