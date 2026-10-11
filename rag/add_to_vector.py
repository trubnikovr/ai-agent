import sys

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from db_vector import vector_store

# --------------------------------------------------
# Документы
# --------------------------------------------------

texts = [
    {
        "text": (
            "Посейдония — древнегреческий город в Южной Италии, "
            "на побережье Тирренского моря."
        ),
        "source": "paestum_history.txt",
    },
    {
        "text": (
            "Римляне захватили город в 273 году до нашей эры "
            "и дали ему название Паестум."
        ),
        "source": "paestum_history.txt",
    },
    {
        "text": (
            "В Паестуме находятся три древних греческих храма "
            "дорического ордера."
        ),
        "source": "paestum_temples.txt",
    },
]

documents = [
    Document(
        page_content=item["text"],
        metadata={"source": item["source"]},
    )
    for item in texts
]

# Разбиваем длинные документы на части.
# Для коротких примеров текст может остаться одним фрагментом.
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)
chunks = splitter.split_documents(documents)
#
# print(chunks)
# sys.exit();


vector_store.add_documents(chunks)

print(f"Добавлено фрагментов: {len(chunks)}")