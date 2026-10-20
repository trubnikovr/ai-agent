from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

import pymupdf
import pytesseract
from PIL import Image
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from stores.db_vector import vector_store


# Получаем существующий PGVector из твоего менеджера.
# Имя коллекции должно совпадать с namespace, который ты используешь.
NAMESPACE = "pdf_ingestion_demo"
pgvector = vector_store.get_store(NAMESPACE)

LANGUAGES = "rus+eng"
MIN_NATIVE_TEXT_CHARS = 40
MIN_OCR_CONFIDENCE = 25.0
MAX_PDF_SIZE_MB = 100
MAX_PAGES = 500


def validate_pdf_path(path: Path) -> None:
    """Проверяет файл до открытия и обработки."""

    if not path.exists():
        raise FileNotFoundError(f"PDF не найден: {path}")

    if not path.is_file():
        raise ValueError(f"Это не файл: {path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Ожидается PDF-файл, получено: {path.suffix}")

    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > MAX_PDF_SIZE_MB:
        raise ValueError(
            f"PDF слишком большой: {size_mb:.1f} МБ; "
            f"лимит примера — {MAX_PDF_SIZE_MB} МБ"
        )


def normalize_text(text: str) -> str:
    """Убирает лишние пробелы и пустые строки."""

    lines = [" ".join(line.split()) for line in text.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def extract_tables_as_text(page: pymupdf.Page) -> list[str]:
    """
    Ищет таблицы на странице и преобразует их в читаемый текст.

    Это зависит от качества PDF и не гарантирует распознавание
    любой сложной или сканированной таблицы.
    """

    if not hasattr(page, "find_tables"):
        return []

    try:
        table_finder = page.find_tables()
    except Exception as exc:
        print(f"Предупреждение: таблицы на странице не извлечены: {exc}")
        return []

    extracted_tables: list[str] = []

    for table_number, table in enumerate(table_finder.tables, start=1):
        rows = table.extract()

        if not rows:
            continue

        cleaned_rows = [
            [normalize_text(str(cell or "")) for cell in row]
            for row in rows
        ]

        # Пропускаем пустую таблицу.
        if not any(any(cell for cell in row) for row in cleaned_rows):
            continue

        lines = [f"Таблица {table_number}:"]

        for row_number, row in enumerate(cleaned_rows, start=1):
            values = [cell for cell in row if cell]
            if values:
                lines.append(f"Строка {row_number}: " + " | ".join(values))

        extracted_tables.append("\n".join(lines))

    return extracted_tables


def run_ocr(page: pymupdf.Page) -> tuple[str, float]:
    """Рендерит страницу и возвращает OCR-текст и среднюю уверенность."""

    pixmap = page.get_pixmap(
        matrix=pymupdf.Matrix(2, 2),
        alpha=False,
    )

    image = Image.frombytes(
        "RGB",
        (pixmap.width, pixmap.height),
        pixmap.samples,
    )

    # image_to_data возвращает распознанные слова и confidence для каждого.
    ocr_data = pytesseract.image_to_data(
        image,
        lang=LANGUAGES,
        output_type=pytesseract.Output.DICT,
    )

    words: list[str] = []
    confidences: list[float] = []

    for word, confidence_text in zip(
        ocr_data["text"],
        ocr_data["conf"],
    ):
        word = normalize_text(word)
        if not word:
            continue

        try:
            confidence = float(confidence_text)
        except (TypeError, ValueError):
            continue

        # Tesseract использует отрицательное значение для слов без оценки.
        if confidence >= 0:
            words.append(word)
            confidences.append(confidence)

    text = " ".join(words)
    average_confidence = (
        sum(confidences) / len(confidences)
        if confidences
        else 0.0
    )

    return normalize_text(text), average_confidence


def extract_page_document(
    page: pymupdf.Page,
    *,
    source: str,
    page_number: int,
) -> Document | None:
    """Извлекает текст страницы; при необходимости запускает OCR."""

    native_text = normalize_text(page.get_text("text"))
    tables = extract_tables_as_text(page)

    content_parts: list[str] = []

    if native_text:
        content_parts.append(native_text)

    # Таблицы добавляем отдельно, чтобы сохранить связи ячеек как текст.
    content_parts.extend(tables)

    extraction_method = "native_text"
    ocr_confidence: float | None = None

    # На сканированной странице native-текста почти нет — запускаем OCR.
    if len(native_text) < MIN_NATIVE_TEXT_CHARS:
        try:
            ocr_text, ocr_confidence = run_ocr(page)
        except pytesseract.TesseractNotFoundError as exc:
            raise RuntimeError(
                "Tesseract не установлен или не найден в PATH. "
                "Установи Tesseract и языковые данные rus/eng."
            ) from exc
        except pytesseract.TesseractError as exc:
            raise RuntimeError(
                f"Ошибка OCR на странице {page_number}: {exc}"
            ) from exc

        if ocr_text:
            content_parts.insert(0, ocr_text)
            extraction_method = "ocr"

    content = "\n\n".join(part for part in content_parts if part).strip()

    if not content:
        print(f"Страница {page_number}: текст не извлечён, пропускаю.")
        return None

    if (
        extraction_method == "ocr"
        and ocr_confidence is not None
        and ocr_confidence < MIN_OCR_CONFIDENCE
    ):
        print(
            f"Предупреждение: низкая уверенность OCR "
            f"на странице {page_number}: {ocr_confidence:.1f}"
        )

    return Document(
        page_content=content,
        metadata={
            "source": source,
            "page": page_number,
            "extraction_method": extraction_method,
            "ocr_confidence": ocr_confidence,
        },
    )


def stable_chunk_id(document: Document, chunk_number: int) -> str:
    """Строит детерминированный ID для чанка."""

    source = str(document.metadata["source"])
    page = int(document.metadata["page"])
    text_hash = hashlib.sha256(
        document.page_content.encode("utf-8")
    ).hexdigest()[:16]

    raw_id = f"{source}:{page}:{chunk_number}:{text_hash}"
    return hashlib.sha256(raw_id.encode("utf-8")).hexdigest()


def ingest_pdf(pdf_path: Path) -> int:
    """Извлекает PDF, разбивает на chunks и добавляет их в PGVector."""

    validate_pdf_path(pdf_path)

    try:
        pdf = pymupdf.open(pdf_path)
    except Exception as exc:
        raise ValueError(f"Не удалось открыть PDF: {exc}") from exc

    with pdf:
        if pdf.is_encrypted:
            raise ValueError("PDF зашифрован; для примера нужен незашифрованный файл")

        if pdf.page_count == 0:
            raise ValueError("В PDF нет страниц")

        if pdf.page_count > MAX_PAGES:
            raise ValueError(
                f"В PDF {pdf.page_count} страниц; "
                f"лимит примера — {MAX_PAGES}"
            )

        page_documents: list[Document] = []

        for page_index in range(pdf.page_count):
            page_number = page_index + 1
            page = pdf.load_page(page_index)

            document = extract_page_document(
                page,
                source=pdf_path.name,
                page_number=page_number,
            )

            if document is not None:
                page_documents.append(document)

    if not page_documents:
        raise ValueError(
            "Не удалось извлечь текст. Проверь качество PDF и настройки OCR."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        add_start_index=True,
    )

    chunks = splitter.split_documents(page_documents)

    if not chunks:
        raise ValueError("После разбиения не получилось ни одного фрагмента")

    # Дополнительная проверка: не отправляем в embeddings пустые куски.
    chunks = [
        chunk
        for chunk in chunks
        if chunk.page_content.strip()
    ]

    ids = [
        stable_chunk_id(chunk, chunk_number)
        for chunk_number, chunk in enumerate(chunks)
    ]

    # Здесь PGVector вызывает embeddings для page_content и сохраняет
    # текст, вектор и metadata в PostgreSQL.
    pgvector.add_documents(
        documents=chunks,
        ids=ids,
    )

    print(f"Страниц с извлечённым текстом: {len(page_documents)}")
    print(f"Фрагментов добавлено в коллекцию {NAMESPACE!r}: {len(chunks)}")

    return len(chunks)


def search_pdf_index(query: str, k: int = 4) -> list[Document]:
    """Ищет похожие фрагменты и проверяет аргументы поиска."""

    query = query.strip()
    if not query:
        raise ValueError("Поисковый запрос не должен быть пустым")

    if k < 1 or k > 20:
        raise ValueError("k должен быть от 1 до 20")

    return pgvector.similarity_search(query, k=k)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(
            "Использование: python -m rag.ingest_pdf /path/to/document.pdf"
        )

    input_pdf = Path(sys.argv[1]).expanduser().resolve()

    try:
        ingest_pdf(input_pdf)

    except Exception as exc:
        raise SystemExit(f"Ошибка обработки: {exc}") from exc