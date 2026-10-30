from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


from ai.ai_client import ai_client
from rag.query_generation.schemas import SQLQuery, DatabaseAnswer
from stores.contract_table_manager import contract_table_manager

sql_llm = ai_client.client.with_structured_output(SQLQuery)
answer_llm = ai_client.client.with_structured_output(DatabaseAnswer)


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

sql_chain = sql_prompt | sql_llm

answer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
Ответь на вопрос только по данным из базы.
Не добавляй условий договора, которых нет в результатах.
Если данных недостаточно, скажи об этом.
""",
    ),
    (
        "human",
        """
Вопрос: {question}

Найденные строки:
{rows}
""",
    ),
])



question = "Может ли исполнитель самостоятельно изменить цену услуг?"
generated_sql = sql_chain.invoke({
    "schema": contract_table_manager.get_schema_description(),
    "question": question,
})
rows = contract_table_manager.execute_select(generated_sql.query)

answer_chain = answer_prompt | answer_llm
answer = answer_chain.invoke({
    "question": question,
    "rows": rows,
})
print(answer)

