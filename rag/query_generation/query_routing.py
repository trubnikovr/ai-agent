# rag/query_generation/query_router.py

from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from ai.ai_client import ai_client

# Готовая SQL-цепочка:
# генерация SELECT → выполнение → формирование ответа.
from rag.query_generation.sql_rag import ask as ask_sql

# Готовая Vector RAG-цепочка:
# embedding → vector/full-text search → RRF
# → similarity postprocessing → ответ.
from rag.query_generation.hybrid_rag import ask as ask_vector


# ==================================================
# Настройки
# ==================================================

VECTOR_RESULT_LIMIT = 3

MINIMUM_VECTOR_SIMILARITY = 0.65


# ==================================================
# Схемы маршрутизатора
# ==================================================

class RouteDecision(BaseModel):
    route: Literal["sql", "vector"] = Field(
        description="Маршрут обработки вопроса"
    )

    reason: str = Field(
        description="Краткое объяснение выбора маршрута"
    )


class RoutedAnswer(BaseModel):
    route: Literal["sql", "vector"]
    route_reason: str
    answer: str
    sources: list[str]


# ==================================================
# Prompt маршрутизатора
# ==================================================

router_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
Ты выбираешь способ поиска ответа во внутренней базе договоров.

Доступны два маршрута.

<SQLRoute>
Выбирай sql для вопросов, требующих точных операций с таблицей:

- подсчитать количество;
- получить список записей;
- отфильтровать по contract_id;
- отфильтровать по section;
- найти конкретный договор или раздел;
- выполнить COUNT;
- получить точные значения из колонок.

Примеры:

- Сколько разделов содержит contract_001?
- Какие разделы есть у contract_001?
- Покажи раздел Цена договора contract_001.
</SQLRoute>

<VectorRoute>
Выбирай vector для вопросов по смыслу текста договора:

- объяснение условий;
- права и обязанности сторон;
- изменение стоимости;
- сроки выполнения;
- ответственность;
- расторжение договора;
- штрафы и неустойки;
- смысловой поиск по тексту.

Примеры:

- Может ли исполнитель самостоятельно изменить цену?
- Что произойдёт при просрочке оплаты?
- Как можно расторгнуть договор?
</VectorRoute>

<Rules>
- выбери только один маршрут;
- не отвечай на сам вопрос;
- route может быть только sql или vector;
- кратко объясни причину выбора.
</Rules>
""",
    ),
    (
        "human",
        """
<Question>
{question}
</Question>
""",
    ),
])


router_chain = (
    router_prompt
    | ai_client.client.with_structured_output(RouteDecision)
)


# ==================================================
# SQL route
# ==================================================

def execute_sql_route(
    question: str,
    decision: RouteDecision,
) -> RoutedAnswer:
    """
    SQL ask() возвращает DatabaseAnswer:

    answer: str
    used_sections: list[str]
    """

    result = ask_sql(question)

    return RoutedAnswer(
        route="sql",
        route_reason=decision.reason,
        answer=result.answer,
        sources=result.used_sections,
    )


# ==================================================
# Vector route
# ==================================================

def execute_vector_route(
    question: str,
    decision: RouteDecision,
) -> RoutedAnswer:
    """
    Vector ask() выполняет:

    1. Создание embedding вопроса.
    2. Векторный поиск.
    3. Полнотекстовый поиск.
    4. Объединение через RRF.
    5. Фильтрацию по vector_similarity.
    6. Формирование ответа по оставшимся документам.

    Возвращает RAGAnswer:

    answer: str
    sources: list[str]
    """

    result = ask_vector(
        question=question,
        k=VECTOR_RESULT_LIMIT,
        minimum_similarity=MINIMUM_VECTOR_SIMILARITY,
    )

    return RoutedAnswer(
        route="vector",
        route_reason=decision.reason,
        answer=result.answer,
        sources=result.sources,
    )


# ==================================================
# Полная цепочка маршрутизации
# ==================================================

def ask(question: str) -> RoutedAnswer:
    """
    Выполняет полный сценарий:

    1. Проверяет вопрос.
    2. Выбирает SQL или Vector route.
    3. Запускает соответствующую цепочку.
    4. Возвращает унифицированный ответ.
    """

    question = question.strip()

    if not question:
        raise ValueError(
            "Вопрос не должен быть пустым"
        )

    # ----------------------------------------------
    # Шаг 1. Выбор маршрута
    # ----------------------------------------------

    decision = router_chain.invoke({
        "question": question,
    })

    print("\nВыбран маршрут:", decision.route)
    print("Причина:", decision.reason)

    # ----------------------------------------------
    # Шаг 2. Выполнение маршрута
    # ----------------------------------------------

    if decision.route == "sql":
        return execute_sql_route(
            question=question,
            decision=decision,
        )

    if decision.route == "vector":
        return execute_vector_route(
            question=question,
            decision=decision,
        )

    raise ValueError(
        f"Неизвестный маршрут: {decision.route}"
    )


# ==================================================
# Проверка
# ==================================================

if __name__ == "__main__":
    questions = [
        # SQL route
        "Сколько разделов содержит договор contract_001?",

        # SQL route
        "Какие разделы есть у договора contract_001?",

        # Vector route
        "Может ли исполнитель самостоятельно изменить цену?",

        # Vector route
        "Что оплачивает заказчик при расторжении договора?",

        # Vector route
        "Какой размер пени установлен за просрочку оплаты?",

        # Может не пройти similarity postprocessor,
        # если такой информации в договорах нет.
        "Какой курс доллара установлен договором?",
    ]

    for current_question in questions:
        print("\n" + "=" * 70)
        print("Вопрос:", current_question)

        response = ask(current_question)

        print("\nМаршрут:")
        print(response.route)

        print("\nПричина:")
        print(response.route_reason)

        print("\nОтвет:")
        print(response.answer)

        print("\nИсточники:")

        if not response.sources:
            print("Источники отсутствуют")
        else:
            for source in response.sources:
                print("-", source)