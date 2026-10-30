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

# Здесь должна быть та же модель, которой заполнена колонка embedding.
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview",
    google_api_key=api_key,
)

llm = ai_client.client


# --------------------------------------------------
# Pydantic-схемы
# --------------------------------------------------

class SearchQuery(BaseModel):
    query: str = Field(
        description="Самостоятельный запрос для поиска по текстам договоров"
    )


class RAGAnswer(BaseModel):
    answer: str = Field(
        description="Ответ на исходный вопрос только по найденным фрагментам"
    )
    sources: list[str] = Field(
        description="Источники ответа в формате contract_id / section"
    )


# --------------------------------------------------
# Шаг 1. Переформулировать вопрос для поиска
# --------------------------------------------------

query_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
Преобразуй вопрос пользователя в самостоятельный запрос
для поиска по текстам договоров.

Правила:
- учитывай историю разговора, если она передана;
- раскрывай ссылки вроде «он», «это», «там»;
- не отвечай на вопрос;
- не добавляй фактов, которых нет в вопросе или истории;
- верни только поле query по заданной схеме.
""",
    ),
    (
        "human",
        """
История разговора:
{history}

Текущий вопрос:
{question}
""",
    ),
])

query_generator = (
    query_prompt
    | llm.with_structured_output(SearchQuery)
)


# --------------------------------------------------
# Шаг 2. Гибридный поиск: векторы + полнотекстовый поиск
# Результаты объединяются через Reciprocal Rank Fusion
# --------------------------------------------------

HYBRID_SEARCH_SQL = text("""
    WITH search_query AS (
        SELECT websearch_to_tsquery('russian', :search_text) AS query
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


def search_sections(search_text: str, k: int = 3):
    """Ищет разделы по смыслу и совпадениям слов."""

    query_embedding = embeddings.embed_query(search_text)

    # PostgreSQL pgvector принимает вектор в виде строки [0.1, 0.2, ...]
    query_vector = "[" + ",".join(
        f"{value:.10f}" for value in query_embedding
    ) + "]"

    # Берём расширенный список кандидатов из обоих поисков.
    candidate_limit = max(k * 5, 20)

    with engine.connect() as connection:
        rows = connection.execute(
            HYBRID_SEARCH_SQL,
            {
                "search_text": search_text,
                "query_vector": query_vector,
                "candidate_limit": candidate_limit,
                "result_limit": k,
            },
        ).mappings().all()

    return rows


# --------------------------------------------------
# Шаг 3. Ответить на исходный вопрос по контексту
# --------------------------------------------------

answer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
Ты отвечаешь на вопрос по найденным фрагментам договоров.

Правила:
- используй только информацию из контекста;
- не додумывай отсутствующие условия;
- если данных недостаточно, прямо скажи об этом;
- в sources укажи только источники из списка разрешённых источников;
- если контекст не содержит ответа, верни пустой список sources.

Разрешённые источники:
{allowed_sources}

Контекст:
{context}
""",
    ),
    (
        "human",
        """
Ответь на исходный вопрос:

{question}
""",
    ),
])

answer_chain = (
    answer_prompt
    | llm.with_structured_output(RAGAnswer)
)


# --------------------------------------------------
# Полный RAG-запрос
# --------------------------------------------------

def ask(
    question: str,
    history: str = "",
    k: int = 3,
) -> RAGAnswer:
    # 1. Получаем самостоятельный поисковый запрос.
    generated_query = query_generator.invoke({
        "question": question,
        "history": history,
    })

    search_text = generated_query.query
    print("Запрос для поиска:", search_text)

    # 2. Ищем подходящие разделы гибридным поиском.
    rows = search_sections(search_text, k=k)

    if not rows:
        return RAGAnswer(
            answer="В базе не найдено подходящих фрагментов договора.",
            sources=[],
        )

    # Показываем, какие строки нашла база.
    for row in rows:
        print(
            f"Найдено: {row['contract_id']} / {row['section']}; "
            f"vector_rank={row['vector_rank']}; "
            f"text_rank={row['text_rank']}; "
            f"rrf_score={row['rrf_score']:.4f}"
        )

    # 3. Готовим источники и контекст для модели ответа.
    allowed_sources = [
        f"{row['contract_id']} / {row['section']}"
        for row in rows
    ]

    context = "\n\n".join(
        f"Источник: {row['contract_id']} / {row['section']}\n"
        f"Текст: {row['content']}"
        for row in rows
    )

    # 4. Отвечаем на исходный вопрос, а не на переписанный поисковый запрос.
    result = answer_chain.invoke({
        "question": question,
        "context": context,
        "allowed_sources": "\n".join(allowed_sources),
    })

    # Не оставляем в sources названия, которых не было среди найденных строк.
    result.sources = [
        source
        for source in result.sources
        if source in allowed_sources
    ]

    return result


# --------------------------------------------------
# Запуск
# --------------------------------------------------

if __name__ == "__main__":
    history = """
Мы обсуждали условие договора: исполнитель вправе изменить стоимость
только по письменному соглашению сторон.
"""

    question = "А он может это сделать без согласия?"

    result = ask(
        question=question,
        history=history,
        k=3,
    )

    print("\nОтвет:")
    print(result.answer)

    print("\nИсточники:")
    for source in result.sources:
        print("-", source)