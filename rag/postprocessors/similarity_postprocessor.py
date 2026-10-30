# rag/postprocessors/retrieval_postprocessor.py

from datetime import datetime, timezone

from pydantic import BaseModel, Field


class RetrievedDocument(BaseModel):
    """
    Документ, полученный из гибридного поиска.
    """

    id: int
    contract_id: str
    section: str
    content: str

    updated_at: datetime | None = None

    vector_rank: int | None = None
    text_rank: int | None = None

    vector_similarity: float | None = None
    keyword_score: float | None = None
    rrf_score: float

    freshness_score: float = 0
    final_score: float = 0


class PostprocessingConfig(BaseModel):
    """
    Настройки retrieval postprocessing.
    """

    minimum_similarity: float = Field(
        default=0.65,
        ge=-1,
        le=1,
    )

    result_limit: int = Field(
        default=3,
        gt=0,
    )

    # Через сколько дней freshness уменьшается вдвое.
    freshness_half_life_days: int = Field(
        default=365,
        gt=0,
    )

    # Веса итогового reranking.
    rrf_weight: float = 0.45
    similarity_weight: float = 0.30
    keyword_weight: float = 0.15
    freshness_weight: float = 0.10


class RetrievalPostprocessor:
    """
    Выполняет фильтрацию и повторное ранжирование
    найденных документов.
    """

    def __init__(
        self,
        config: PostprocessingConfig | None = None,
    ):
        self.config = (
            config
            if config is not None
            else PostprocessingConfig()
        )

        self._validate_weights()

    def _validate_weights(self) -> None:
        total_weight = (
            self.config.rrf_weight
            + self.config.similarity_weight
            + self.config.keyword_weight
            + self.config.freshness_weight
        )

        if abs(total_weight - 1.0) > 0.0001:
            raise ValueError(
                "Сумма весов postprocessing должна быть равна 1"
            )

    def process(
        self,
        rows: list[dict],
    ) -> list[RetrievedDocument]:
        """
        Полный postprocessing:

        1. Pydantic-валидация.
        2. Similarity/keyword filtering.
        3. Нормализация оценок.
        4. Freshness score.
        5. Итоговый reranking.
        6. Top K.
        """

        documents = [
            RetrievedDocument.model_validate(row)
            for row in rows
        ]

        documents = self._filter_documents(
            documents
        )

        if not documents:
            return []

        self._calculate_final_scores(
            documents
        )

        documents.sort(
            key=lambda document: document.final_score,
            reverse=True,
        )

        return documents[
            :self.config.result_limit
        ]

    def _filter_documents(
        self,
        documents: list[RetrievedDocument],
    ) -> list[RetrievedDocument]:
        """
        Документ сохраняется, если:

        - vector similarity выше порога;
        ИЛИ
        - документ найден полнотекстовым поиском.

        Точное keyword-совпадение может спасти документ,
        даже если similarity немного ниже порога.
        """

        filtered_documents: list[
            RetrievedDocument
        ] = []

        for document in documents:
            passed_similarity = (
                document.vector_similarity is not None
                and document.vector_similarity
                >= self.config.minimum_similarity
            )

            passed_keyword = (
                document.text_rank is not None
                and document.keyword_score is not None
                and document.keyword_score > 0
            )

            if passed_similarity or passed_keyword:
                filtered_documents.append(
                    document
                )

        return filtered_documents

    def _calculate_final_scores(
        self,
        documents: list[RetrievedDocument],
    ) -> None:
        """
        Нормализует разные оценки и рассчитывает final_score.
        """

        maximum_rrf = max(
            document.rrf_score
            for document in documents
        )

        maximum_keyword_score = max(
            (
                document.keyword_score or 0
                for document in documents
            ),
            default=0,
        )

        for document in documents:
            normalized_rrf = self._normalize(
                value=document.rrf_score,
                maximum=maximum_rrf,
            )

            normalized_similarity = self._normalize_similarity(
                document.vector_similarity
            )

            normalized_keyword = self._normalize(
                value=document.keyword_score or 0,
                maximum=maximum_keyword_score,
            )

            document.freshness_score = (
                self._calculate_freshness(
                    document.updated_at
                )
            )

            document.final_score = (
                normalized_rrf
                * self.config.rrf_weight

                + normalized_similarity
                * self.config.similarity_weight

                + normalized_keyword
                * self.config.keyword_weight

                + document.freshness_score
                * self.config.freshness_weight
            )

    @staticmethod
    def _normalize(
        value: float,
        maximum: float,
    ) -> float:
        """
        Приводит оценку к диапазону 0..1.
        """

        if maximum <= 0:
            return 0

        return min(
            max(value / maximum, 0),
            1,
        )

    @staticmethod
    def _normalize_similarity(
        similarity: float | None,
    ) -> float:
        """
        Ограничивает cosine similarity диапазоном 0..1.

        Отрицательная similarity не даёт бонуса.
        """

        if similarity is None:
            return 0

        return min(
            max(similarity, 0),
            1,
        )

    def _calculate_freshness(
        self,
        updated_at: datetime | None,
    ) -> float:
        """
        Экспоненциальное уменьшение freshness.

        Если half_life = 365:
        - сегодня: 1.0;
        - через 365 дней: 0.5;
        - через 730 дней: 0.25.
        """

        if updated_at is None:
            return 0

        if updated_at.tzinfo is None:
            updated_at = updated_at.replace(
                tzinfo=timezone.utc
            )

        now = datetime.now(
            timezone.utc
        )

        age = now - updated_at

        age_days = max(
            age.total_seconds() / 86400,
            0,
        )

        return 0.5 ** (
            age_days
            / self.config.freshness_half_life_days
        )