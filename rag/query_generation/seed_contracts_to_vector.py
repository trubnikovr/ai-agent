from langchain_classic.chains.query_constructor.schema import AttributeInfo
from langchain_core.documents import Document

from config import configure_langchain
from stores.db_vector import vector_store

configure_langchain()

NAMESPACE = "contract_rules_chapter_10_self_query"
ID_KEY = "contract_id"

pgvector = vector_store.get_store(NAMESPACE)


sections = [
    {
        "contract_id": "contract_001",
        "section": "Цена",
        "content": (
            "Цена услуг составляет 10 000 000 сум и фиксирована "
            "на весь срок действия договора. Исполнитель вправе "
            "изменить стоимость только по письменному соглашению сторон."
        ),
    },
    {
        "contract_id": "contract_001",
        "section": "Сроки",
        "content": (
            "Исполнитель начинает работу после получения аванса "
            "и необходимых материалов. Срок выполнения составляет "
            "30 календарных дней. Если заказчик задерживает передачу "
            "материалов или согласование результата, срок продлевается "
            "на период задержки."
        ),
    },
    {
        "contract_id": "contract_001",
        "section": "Приёмка",
        "content": (
            "Заказчик в течение пяти рабочих дней после получения "
            "результата подписывает акт или направляет письменный "
            "мотивированный отказ с перечнем замечаний."
        ),
    },
    {
        "contract_id": "contract_001",
        "section": "Ответственность",
        "content": (
            "За просрочку оплаты заказчик уплачивает пеню 0,1 процента "
            "от просроченной суммы за каждый день. Общая пеня не может "
            "превышать 10 процентов просроченной суммы."
        ),
    },
    {
        "contract_id": "contract_001",
        "section": "Расторжение",
        "content": (
            "Каждая сторона может предложить расторжение договора, "
            "направив письменное уведомление не менее чем за 15 дней. "
            "Заказчик оплачивает оказанные услуги и заранее согласованные "
            "расходы исполнителя."
        ),
    },
]


documents = [
    Document(
        page_content=section['content'],
        metadata={
            "contract_id": section["contract_id"],
            "section": section["section"],
            "source": "contract_001.txt",
        },
    )
    for section in sections
]

document_ids = [
    f'{section["contract_id"]}:{section["section"]}'
    for section in sections
]
pgvector.delete(ids=document_ids)
pgvector.add_documents(
    documents=documents,
    ids=document_ids,
)

print(f"В pgvector загружено документов: {len(documents)}")
