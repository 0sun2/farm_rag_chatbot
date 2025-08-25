import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# 텍스트 정제 함수
def clean_text(text: str) -> str:
    lines = text.splitlines()
    cleaned = [
        line for line in lines
        if not line.strip().isdigit()                   # 숫자만 있는 줄 제거
        and not line.strip().lower().startswith("chapter")  # 목차 제거
        and len(line.strip()) > 10                      # 너무 짧은 줄 제거
        and not any(x in line for x in ["목차", "작형", "부록", "특집"])  # 장 제목 제거
    ]
    return "\n".join(cleaned)

# PDF 경로 설정
pdf_folder = "C:/Users/lys/Desktop/RAG/농업pdf"
pdf_files = ["감자.pdf", "농업기술길잡이40_딸기.pdf"]

all_chunks = []
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

for filename in pdf_files:
    full_path = os.path.join(pdf_folder, filename)
    loader = PyMuPDFLoader(full_path)
    docs = loader.load()

    # 정제 후 청크 분할
    for doc in docs:
        doc.page_content = clean_text(doc.page_content)

    chunks = splitter.split_documents(docs)
    all_chunks.extend(chunks)

# 임베딩 생성기
embedding = HuggingFaceEmbeddings(model_name="jhgan/ko-sbert-nli")

# FAISS DB 생성 및 저장
db = FAISS.from_documents(all_chunks, embedding)
db.save_local("strawberry_potato_FAISS_DB")
