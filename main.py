from config import configure_langchain
from rag.query_generation.seed_contracts_to_vector import NAMESPACE
from stores.db_vector import vector_store
from stores.contract_table_manager import contract_table_manager

configure_langchain()

pgvector = vector_store.get_store(NAMESPACE)

try:
    count = contract_table_manager.fill_embeddings(
        embeddings=pgvector.embeddings,
    )
    print(f"Embedding созданы для разделов: {count}")
finally:
    contract_table_manager.close()