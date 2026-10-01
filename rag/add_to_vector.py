from langchain_core.documents import Document

from db_vector import vector_store

# --------------------------------------------------
# Документы
# --------------------------------------------------

documents = [
    Document(
        page_content=(
            "Изменение стоимости услуг допускается только "
            "по письменному соглашению обеих сторон."
        ),
        metadata={
            "source": "contract_rules",
            "section": "price",
            "document_id": 1,
        },
    ),
    Document(
        page_content=(
            "Срок оплаты составляет 10 рабочих дней "
            "после подписания акта выполненных работ."
        ),
        metadata={
            "source": "contract_rules",
            "section": "payment",
            "document_id": 1,
        },
    ),
    Document(
        page_content=(
            "За каждый календарный день просрочки оплаты "
            "заказчик уплачивает неустойку в размере "
            "0.1% от суммы задолженности."
        ),
        metadata={
            "source": "contract_rules",
            "section": "penalty",
            "document_id": 1,
        },
    ),
]

document_ids = [
    "contract-rule-price",
    "contract-rule-payment",
    "contract-rule-penalty",
]


vector_store.add_documents(
    documents=documents,
    ids=document_ids,
)
