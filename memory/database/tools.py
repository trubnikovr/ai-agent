from memory.database.agent_context import UserInfo, AgentContext
from langchain.tools import ToolRuntime, tool



@tool
def save_user_info(
    user_info: UserInfo,
    runtime: ToolRuntime[AgentContext],
) -> str:
    """Сохраняет имя и профессию пользователя."""

    user_id = runtime.context.user_id

    runtime.store.put(
        ("users",),
        user_id,
        dict(user_info),
    )

    return "Информация о пользователе сохранена"


@tool
def get_user_info(
    runtime: ToolRuntime[AgentContext],
) -> dict | str:
    """Возвращает сохранённую информацию о пользователе."""

    user_id = runtime.context.user_id

    item = runtime.store.get(
        ("users",),
        user_id,
    )

    if item is None:
        return "Информация о пользователе не найдена"

    return item.value
