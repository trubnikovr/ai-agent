from langchain.agents import create_agent
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
from dotenv import dotenv_values
from ai.ai_client import ai_client
from memory.database.agent_context import AgentContext
from memory.database.tools import save_user_info, get_user_info

# --------------------------------------------------
# Настройки
# --------------------------------------------------

config = dotenv_values(".env")

database_url = config.get("DATABASE_URL")

if not database_url:
    raise ValueError("DATABASE_URL не найден в .env")


llm = ai_client.client



# --------------------------------------------------
# PostgreSQL-память
# --------------------------------------------------

with (
    PostgresStore.from_conn_string(database_url) as store,
    PostgresSaver.from_conn_string(database_url) as checkpointer,
):
    # Создаёт служебные таблицы LangGraph.
    # Выполняется при инициализации приложения, а не на каждый запрос.
    store.setup()
    checkpointer.setup()

    agent = create_agent(
        model=llm,
        tools=[
            save_user_info,
            get_user_info,
        ],
        store=store,
        checkpointer=checkpointer,
        context_schema=AgentContext,
        system_prompt="""
Ты персональный помощник.

Правила:
- если пользователь сообщает имя и профессию,
  сохрани их через save_user_info;
- если пользователь спрашивает информацию о себе,
  используй get_user_info;
- не придумывай данные пользователя.
""",
    )

    user_context = AgentContext(
        user_id="user-123",
    )

    first_thread = {
        "configurable": {
            "thread_id": "dialog-001",
        }
    }

    # Первый диалог
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Меня зовут Рустам, "
                        "я backend-разработчик"
                    ),
                }
            ]
        },
        config=first_thread,
        context=user_context,
    )

    print(result["messages"][-1].content)

    # Продолжаем тот же диалог
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Что я только что тебе сообщил?",
                }
            ]
        },
        config=first_thread,
        context=user_context,
    )

    print(result["messages"][-1].content)

    # Новый диалог того же пользователя
    second_thread = {
        "configurable": {
            "thread_id": "dialog-002",
        }
    }

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Что ты знаешь обо мне?",
                }
            ]
        },
        config=second_thread,
        context=user_context,
    )

    print(result["messages"][-1].content)