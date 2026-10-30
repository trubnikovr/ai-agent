from collections.abc import Mapping
from typing import Any


def filter_by_similarity(
    rows: list[Mapping[str, Any]],
    minimum_similarity: float = 0.65,
) -> list[Mapping[str, Any]]:
    """
    Оставляет только документы, близость которых
    не ниже minimum_similarity.
    """

    filtered_rows = []

    for row in rows:
        similarity = row.get("vector_similarity")

        if similarity is None:
            continue

        if float(similarity) >= minimum_similarity:
            filtered_rows.append(row)

    return filtered_rows