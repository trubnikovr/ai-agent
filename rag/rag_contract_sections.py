from dotenv import dotenv_values
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, text

from ai.ai_client import ai_client
from config import settings


# --------------------------------------------------
# 1. Подключение к PostgreSQL и Gemini Embeddings
# --------------------------------------------------

config = dotenv_values(".env")

api_key = config.get("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY не найден в .env")

engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
)

# Важно: используй ту же embedding-модель,
# которой заполнял колонку embedding.
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview",
    google_api_key=api_key,
)


# --------------------------------------------------
# 2. Pydantic-схема ответа
# --------------------------------------------------

class RAGAnswer(BaseModel):
    answer: str = Field(
        description="Ответ на вопрос, основанный только на найденных фрагментах",
    )
    sources: list[str] = Field(
        description="Источники ответа в формате contract_id / section",
    )


# Просим Gemini вернуть экземпляр RAGAnswer
structured_llm = ai_client.client.with_structured_output(RAGAnswer)


# --------------------------------------------------
# 3. Prompt для ответа по найденному контексту
# --------------------------------------------------

answer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
Ты отвечаешь на вопрос по фрагментам договоров.

Правила:
- используй только информацию из контекста;
- не додумывай отсутствующие условия;
- если контекста недостаточно, скажи об этом;
- в sources укажи только источники из контекста.

Контекст:
{context}
""",
    ),
    (
        "human",
        "{question}",
    ),
])

answer_chain = answer_prompt | structured_llm


# --------------------------------------------------
# 4. Семантический поиск и формирование ответа
# --------------------------------------------------

def ask(question: str, k: int = 3) -> RAGAnswer:
    # Превращаем вопрос в embedding-вектор.
    # Это вызов embedding-модели, а не обычный текстовый запрос Gemini.
    query_embedding = embeddings.embed_query(question)

    # Преобразуем список чисел в формат PostgreSQL vector: [0.1, 0.2, ...]
    query_vector = "[" + ",".join(
        f"{value:.10f}" for value in query_embedding
    ) + "]"

    # PostgreSQL ищет ближайшие к вопросу векторы.
    # Оператор <=> означает cosine distance: меньше — ближе.
    search_sql = text("""
        SELECT
            id,
            contract_id,
            section,
            content,
            1 - (embedding <=> CAST(:query_vector AS vector)) AS similarity
        FROM rag_contract_sections
        WHERE embedding IS NOT NULL
        ORDER BY embedding <=> CAST(:query_vector AS vector)
        LIMIT :limit
    """)

    with engine.connect() as connection:
        rows = connection.execute(
            search_sql,
            {
                "query_vector": query_vector,
                "limit": k,
            },
        ).mappings().all()

    # Если похожих строк нет, не отправляем модели пустой контекст.
    if not rows:
        return RAGAnswer(
            answer="В базе не найдено подходящих фрагментов договора.",
            sources=[],
        )

    # Собираем найденные фрагменты в контекст для Gemini.
    context_parts = []

    for row in rows:
        source_id = f"{row['contract_id']} / {row['section']}"

        context_parts.append(
            f"""
Источник: {source_id}
Текст: {row['content']}
Сходство: {row['similarity']:.3f}
""".strip()
        )

    context = "\n\n".join(context_parts)

    # Gemini формулирует ответ только по найденному контексту.
    return answer_chain.invoke({
        "question": question,
        "context": context,
    })


# --------------------------------------------------
# 5. Запуск
# --------------------------------------------------

if __name__ == "__main__":
    question = "Может ли исполнитель самостоятельно изменить цену услуг?"

    result = ask(question, k=3)

    print("Ответ:")
    print(result.answer)

    print("\nИсточники:")
    for source in result.sources:
        print("-", source)