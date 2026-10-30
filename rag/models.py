from datetime import datetime

from sqlalchemy import BigInteger, Text, Identity, UniqueConstraint, DateTime, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class ContractSection(Base):
    __tablename__ = "rag_contract_sections"
    __table_args__ = (
        UniqueConstraint("contract_id", "section"),
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
        comment="Дата последнего изменения раздела",
    )

    id: Mapped[int] = mapped_column(
        BigInteger,
        Identity(always=True),
        primary_key=True,
        comment="Уникальный идентификатор записи",
    )
    contract_id: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="Идентификатор договора",
    )
    section: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="Название раздела договора, например «Цена» или «Расторжение»",
    )
    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="Текст условий договора из этого раздела",
    )



