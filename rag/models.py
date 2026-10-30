from sqlalchemy import BigInteger, Text, Identity, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class ContractSection(Base):
    __tablename__ = "rag_contract_sections"
    __table_args__ = (
        UniqueConstraint("contract_id", "section"),
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


