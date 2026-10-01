from dotenv import dotenv_values

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector

# --------------------------------------------------
# Настройки
# --------------------------------------------------

config = dotenv_values(".env")

api_key = config.get("GEMINI_API_KEY")
pgvector_url = config.get("PGVECTOR_URL")

if not api_key:
    raise ValueError("GEMINI_API_KEY не найден")

if not pgvector_url:
    raise ValueError("PGVECTOR_URL не найден")


# --------------------------------------------------
# Embeddings
# --------------------------------------------------

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview",
    google_api_key=api_key,
)

# --------------------------------------------------
# PGVector
# --------------------------------------------------

vector_store = PGVector(
    embeddings=embeddings,
    collection_name="contract_rules",
    connection=pgvector_url,
    use_jsonb=True,
)
