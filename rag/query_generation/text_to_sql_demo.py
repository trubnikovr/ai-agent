from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field

from ai.ai_client import ai_client
from stores.contract_table_manager import contract_table_manager

class SQLQuery(BaseModel):
    query: str = Field(
        description="Один SQL SELECT-запрос к таблице rag_contract_sections"
    )

sql_llm = ai_client.client.with_structured_output(SQLQuery)

sql_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
Ты преобразуешь вопрос пользователя в один SQL-запрос для PostgreSQL.

Правила:
- создай только SELECT;
- используй только таблицу rag_contract_sections;
- используй только колонки из предоставленной схемы;
- выбери только id, contract_id, section, content;
- добавь LIMIT 10;
- верни только SQL, без Markdown и пояснений.
""",
    ),
    (
        "human",
        """
Схема:
{schema}

Вопрос:
{question}
""",
    ),
])

sql_chain = sql_prompt | ai_client.client | StrOutputParser()

question = "Может ли исполнитель самостоятельно изменить цену услуг?"
generated_sql = sql_chain.invoke({
    "schema": contract_table_manager.get_schema_description(),
    "question": question,
})


rows = contract_table_manager.execute_select(generated_sql)
print(rows)