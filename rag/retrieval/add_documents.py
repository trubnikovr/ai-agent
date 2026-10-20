from langchain_core.documents import Document

from config import configure_langchain
from stores.db_vector import vector_store


NAMESPACE = "contract_rules_multi_query_demo"


def main() -> None:
    configure_langchain()

    # Получаем отдельную PGVector-коллекцию.
    # Если её ещё нет, ваш get_store может создать её при первом обращении.
    pgvector = vector_store.get_store(NAMESPACE)

    documents = [
        Document(
            page_content=(
                "Каждая сторона вправе предложить расторжение договора, "
                "направив другой стороне письменное уведомление "
                "не менее чем за 15 календарных дней."
            ),
            metadata={
                "source": "contract_001.txt",
                "section": "Расторжение",
            },
        ),
        Document(
            page_content=(
                "При расторжении заказчик оплачивает фактически оказанные "
                "услуги и расходы исполнителя, которые были "
                "предварительно согласованы сторонами."
            ),
            metadata={
                "source": "contract_001.txt",
                "section": "Расчёты при расторжении",
            },
        ),
        Document(
            page_content=(
                "Заказчик оплачивает счёт в течение пяти рабочих дней "
                "после его получения. Обязательство по оплате считается "
                "исполненным в день поступления денег на счёт исполнителя."
            ),
            metadata={
                "source": "contract_001.txt",
                "section": "Порядок оплаты",
            },
        ),
        Document(
            page_content=(
                "Исполнитель вправе изменить стоимость услуг только "
                "по письменному соглашению сторон. Уведомление "
                "по электронной почте само по себе не считается "
                "согласием заказчика на изменение цены."
            ),
            metadata={
                "source": "contract_001.txt",
                "section": "Цена",
            },
        ),
        Document(
            page_content=(
                "За нарушение срока оплаты заказчик уплачивает пеню "
                "в размере 0,1 процента от просроченной суммы "
                "за каждый день просрочки. Общая сумма пени "
                "не может превышать 10 процентов от просроченной суммы."
            ),
            metadata={
                "source": "contract_001.txt",
                "section": "Ответственность",
            },
        ),
    ]

    # Постоянные ID позволяют при повторном запуске обновлять эти записи,
    # а не создавать новые копии.
    ids = [
        "contract-001-termination",
        "contract-001-termination-settlement",
        "contract-001-payment",
        "contract-001-price",
        "contract-001-liability",
    ]

    pgvector.add_documents(
        documents=documents,
        ids=ids,
    )

    print(f"Добавлено документов: {len(documents)}")
    print(f"Коллекция: {NAMESPACE}")


if __name__ == "__main__":
    main()