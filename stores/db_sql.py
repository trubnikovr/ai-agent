from langchain_classic.storage import create_kv_docstore
from langchain_community.storage.sql import SQLStore as LangChainSQLStore
from config import settings


class SQLStore:
    """Factory/cache for SQL-backed document stores, keyed by namespace."""

    def __init__(self, db_url: str):
        if not db_url:
            raise ValueError("db_url обязателен")

        self._db_url = db_url
        self._stores: dict[str, LangChainSQLStore] = {}

    def get_store(self, namespace: str, **kwargs) -> LangChainSQLStore:
        if not namespace:
            raise ValueError("namespace обязателен")

        if namespace not in self._stores:
            self._stores[namespace] = LangChainSQLStore(
                namespace=namespace,
                db_url=self._db_url,
                **kwargs,
            )

        return self._stores[namespace]

    def get_kv_docstore(self, *, namespace: str, **kwargs):
        return create_kv_docstore(self.get_store(namespace, **kwargs))


    def create_schema(self, namespace: str, **kwargs) -> None:
        """Create the SQLStore table for the given namespace if it is missing."""
        self.get_store(namespace, **kwargs).create_schema()


def create_sql_store(connection: str) -> SQLStore:
    return SQLStore(db_url=connection)


sql_store = create_sql_store(connection=settings.database_url)
