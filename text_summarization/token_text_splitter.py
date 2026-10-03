from pathlib import Path
from langchain_text_splitters import TokenTextSplitter

splitter = TokenTextSplitter(
    chunk_size=3000,
    chunk_overlap=100,
)

file_path = Path(__file__).parent / "Moby-Dick.txt"

with file_path.open("r", encoding="utf-8") as file:
    document_text = file.read()

chunks = splitter.split_text(document_text)

print(f"Получилось частей: {len(chunks)}")
print(chunks[0][:500])