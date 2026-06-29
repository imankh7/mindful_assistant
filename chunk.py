from langchain_text_splitters import RecursiveCharacterTextSplitter

with open("data/processed/gad.txt", "r", encoding="utf-8") as f:
    text = f.read()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

chunks = splitter.split_text(text)
print(f"Document length: {len(text)} characters")
print(f"Number of chunks: {len(chunks)}")
print("-" * 60)
for i, chunk in enumerate(chunks[:3]):
    print(f"\n--- Chunk {i} ({len(chunk)} characters) ---")
    print(chunk)
