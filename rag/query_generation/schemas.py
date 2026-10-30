from pydantic import BaseModel, Field
class SQLQuery(BaseModel):
    query: str = Field(
        description="Один SQL SELECT-запрос к таблице rag_contract_sections"
    )


class DatabaseAnswer(BaseModel):
    answer: str = Field(description="Ответ пользователю по найденным строкам")
    used_sections: list[str] = Field(
        description="Разделы договора, на которых основан ответ"
    )
