from pgvector.sqlalchemy import Vector
from sqlalchemy import create_engine, select, text, bindparam
from sqlalchemy.orm import Session

from config import settings
from rag.models import Base, ContractSection


class ContractTableManager:
    def __init__(self, database_url: str):
        if not database_url:
            raise ValueError("database_url обязателен")

        self._engine = create_engine(
            database_url,
            pool_pre_ping=True,
        )

    def create_tables(self) -> None:
        Base.metadata.create_all(self._engine)



    def prepare_embeddings(self, embeddings) -> int:
        # Узнаём размерность, вызвав ту же модель, которая будет кодировать тексты.
        sample_vector = embeddings.embed_query("Проверка размерности")
        dimensions = len(sample_vector)

        with self._engine.begin() as connection:
            # Добавляем колонку. Повторный запуск безопасен.
            connection.execute(text(
                f"""
                ALTER TABLE rag_contract_sections
                ADD COLUMN IF NOT EXISTS embedding vector({dimensions})
                """
            ))

            rows = connection.execute(text("""
                                           SELECT id, content
                                           FROM rag_contract_sections
                                           WHERE embedding IS NULL
                                           """)).mappings().all()

            for row in rows:
                vector = embeddings.embed_query(row["content"])

                connection.execute(
                    text("""
                         UPDATE rag_contract_sections
                         SET embedding = :embedding
                         WHERE id = :id
                         """).bindparams(
                        # Типизируем параметр как pgvector, чтобы SQLAlchemy
                        # корректно сериализовал Python-список в vector.
                        __import__("sqlalchemy").bindparam(
                            "embedding",
                            type_=Vector(dimensions),
                        )
                    ),
                    {"id": row["id"], "embedding": vector},
                )

        return len(rows)


    def fill_embeddings(self, embeddings) -> int:
        dimensions = len(embeddings.embed_query("dimension check"))

        with self._engine.begin() as connection:
            rows = connection.execute(text("""
                                           SELECT id, content
                                           FROM rag_contract_sections
                                           WHERE embedding IS NULL
                                           ORDER BY id
                                           """)).mappings().all()

            if not rows:
                return 0

            texts = [row["content"] for row in rows]
            vectors = embeddings.embed_documents(texts)

            update_query = text("""
                                UPDATE rag_contract_sections
                                SET embedding = :embedding
                                WHERE id = :id
                                """).bindparams(
                bindparam("embedding", type_=Vector(dimensions))
            )

            for row, vector in zip(rows, vectors):
                connection.execute(
                    update_query,
                    {"id": row["id"], "embedding": vector},
                )

        return len(rows)

    def save_sections(self, sections: list[dict[str, str]]) -> None:
        with Session(self._engine) as session:
            for item in sections:
                existing = session.scalar(
                    select(ContractSection).where(
                        ContractSection.contract_id == item["contract_id"],
                        ContractSection.section == item["section"],
                    )
                )

                if existing:
                    existing.content = item["content"]
                else:
                    session.add(ContractSection(**item))

            session.commit()

    def get_all_sections(self) -> list[ContractSection]:
        with Session(self._engine) as session:
            statement = select(ContractSection).order_by(ContractSection.id)
            return list(session.scalars(statement))

    def execute_select(self, query: str) -> list[dict]:
        """Выполняет SELECT, сгенерированный для учебного примера."""

        if not query.lstrip().upper().startswith("SELECT"):
            raise ValueError("Разрешены только SELECT-запросы")

        with self._engine.connect() as connection:
            result = connection.execute(text(query))
            return [dict(row._mapping) for row in result]

    def close(self) -> None:
        self._engine.dispose()

    def get_schema_description(self) -> str:
        table = ContractSection.__table__

        column_descriptions = "\n".join(
            f"- {column.name} ({column.type}): "
            f"{column.comment or 'описание отсутствует'}"
            for column in table.columns
        )

        return f"Таблица: {table.name}\nКолонки:\n{column_descriptions}"


contract_table_manager = ContractTableManager(
    database_url=settings.database_url,
)