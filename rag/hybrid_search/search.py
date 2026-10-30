from dotenv import dotenv_values
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, text

from ai.ai_client import ai_client
from config import settings


# --------------------------------------------------
# Подключения
# --------------------------------------------------

config = dotenv_values(".env")

api_key = config.get("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY не найден в .env")

engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
)

# Используй ту же модель, которой заполнял колонку embedding.
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview",
    google_api_key=api_key,
)


# --------------------------------------------------
# Гибридный поиск: векторы + полнотекстовый поиск + RRF
# --------------------------------------------------

HYBRID_SEARCH_SQL = text("""
    WITH search_query AS (
        SELECT websearch_to_tsquery('russian', :question) AS query
    ),
    vector_results AS (
        SELECT
            id,
            ROW_NUMBER() OVER (
                ORDER BY embedding <=> CAST(:query_vector AS vector)
            ) AS rank
        FROM rag_contract_sections
        WHERE embedding IS NOT NULL
        ORDER BY embedding <=> CAST(:query_vector AS vector)
        LIMIT :candidate_limit
    ),
    text_results AS (
        SELECT
            section.id,
            ROW_NUMBER() OVER (
                ORDER BY ts_rank_cd(
                    to_tsvector('russian', section.content),
                    search_query.query
                ) DESC
            ) AS rank
        FROM rag_contract_sections AS section
        CROSS JOIN search_query
        WHERE to_tsvector('russian', section.content) @@ search_query.query
        ORDER BY ts_rank_cd(
            to_tsvector('russian', section.content),
            search_query.query
        ) DESC
        LIMIT :candidate_limit
    ),
    combined AS (
        SELECT
            id,
            rank AS vector_rank,
            NULL::bigint AS text_rank,
            1.0 / (60 + rank) AS rrf_score
        FROM vector_results

        UNION ALL

        SELECT
            id,
            NULL::bigint AS vector_rank,
            rank AS text_rank,
            1.0 / (60 + rank) AS rrf_score
        FROM text_results
    ),
    ranked_results AS (
        SELECT
            id,
            MAX(vector_rank) AS vector_rank,
            MAX(text_rank) AS text_rank,
            SUM(rrf_score) AS rrf_score
        FROM combined
        GROUP BY id
    )
    SELECT
        section.id,
        section.contract_id,
        section.section,
        section.content,
        ranked_results.vector_rank,
        ranked_results.text_rank,
        ranked_results.rrf_score
    FROM ranked_results
    JOIN rag_contract_sections AS section
        ON section.id = ranked_results.id
    ORDER BY ranked_results.rrf_score DESC
    LIMIT :result_limit
""")


def search_sections(question: str, k: int = 3):
    """Возвращает наиболее подходящие разделы, не вызывая чат-модель."""

    query_embedding = embeddings.embed_query(question)
    query_vector = "[" + ",".join(
        f"{value:.10f}" for value in query_embedding
    ) + "]"

    # Берём кандидатов с запасом, затем объединяем их результаты и оставляем k.
    candidate_limit = max(k * 5, 20)

    with engine.connect() as connection:
        rows = connection.execute(
            HYBRID_SEARCH_SQL,
            {
                "question": question,
                "query_vector": query_vector,
                "candidate_limit": candidate_limit,
                "result_limit": k,
            },
        ).mappings().all()

    return rows


# --------------------------------------------------
# Pydantic-схема ответа
# --------------------------------------------------

class RAGAnswer(BaseModel):
    answer: str = Field(
        description="Ответ на вопрос только по переданным фрагментам",
    )
    sources: list[str] = Field(
        description="Источники в формате contract_id / section",
    )


# --------------------------------------------------
# Prompt и структурированный ответ Gemini
# --------------------------------------------------

answer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
Ты отвечаешь на вопрос по найденным фрагментам договора.

Правила:
- используй только информацию из контекста;
- не додумывай отсутствующие условия;
- если данных недостаточно, прямо скажи об этом;
- в sources укажи только источники, перечисленные в контексте.

Контекст:
{context}
""",
    ),
    ("human", "{question}"),
])

answer_chain = (
    answer_prompt
    | ai_client.client.with_structured_output(RAGAnswer)
)


# --------------------------------------------------
# Полный RAG-запрос
# --------------------------------------------------

def ask(question: str, k: int = 3) -> RAGAnswer:
    rows = search_sections(question, k=k)

    # Сначала можно посмотреть, что именно нашёл гибридный поиск.
    for row in rows:
        print(
            f"Найден раздел: {row['contract_id']} / {row['section']} "
            f"(vector rank={row['vector_rank']}, "
            f"text rank={row['text_rank']}, "
            f"RRF={row['rrf_score']:.4f})"
        )

    if not rows:
        return RAGAnswer(
            answer="В базе не найдено подходящих фрагментов.",
            sources=[],
        )

    context_parts = []

    for row in rows:
        source = f"{row['contract_id']} / {row['section']}"
        context_parts.append(
            f"Источник: {source}\n"
            f"Текст: {row['content']}"
        )

    context = "\n\n".join(context_parts)

    return answer_chain.invoke({
        "question": question,
        "context": context,
    })


# --------------------------------------------------
# Запуск
# --------------------------------------------------

if __name__ == "__main__":
    question = "Может ли исполнитель самостоятельно изменить цену услуг?"

    result = ask(question, k=3)

    print("\nОтвет:")
    print(result.answer)

    print("\nИсточники:")
    for source in result.sources:
        print("-", source)