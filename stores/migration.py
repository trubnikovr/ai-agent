from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore

from config import settings
from stores.db_sql import sql_store

SQL_STORE_NAMESPACES = ("contract_rules_parents",)


def migrate() -> None:
    """Create required PostgreSQL tables for LangChain and LangGraph."""
    for namespace in SQL_STORE_NAMESPACES:
        sql_store.create_schema(namespace)

    with (
        PostgresStore.from_conn_string(settings.database_url) as store,
        PostgresSaver.from_conn_string(settings.database_url) as checkpointer,
    ):
        store.setup()
        checkpointer.setup()


if __name__ == "__main__":
    migrate()
