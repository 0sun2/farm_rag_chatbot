import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever, BM25Retriever
import pickle

# --- 1. 문서 로딩 및 기본 분할 ---
pdf_folder = "C:/Users/lys/Desktop/RAG/농업pdf"
pdf_files = ["감자.pdf", "농업기술길잡이40_딸기.pdf"]

all_docs = []
for filename in pdf_files:
    full_path = os.path.join(pdf_folder, filename)
    loader = PyMuPDFLoader(full_path)
    docs = loader.load()
    # 파일 이름을 메타데이터에 추가하여 출처 명시
    for doc in docs:
        doc.metadata['source'] = filename
    all_docs.extend(docs)

print(f"총 {len(all_docs)}개의 페이지를 로딩했습니다.")


#  2. ParentDocumentRetriever


# 자식 청크를 위한 splitter (검색의 정확도를 위해 더 작게 설정)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)

# 부모 청크를 위한 splitter (더 넓은 맥락을 제공하기 위해 약간 더 크게 설정)
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=100)

# 임베딩 모델 로딩
embedding = HuggingFaceEmbeddings(model_name="jhgan/ko-sbert-nli")

# 벡터 DB와 부모 문서 저장소 설정
vectorstore = FAISS.from_documents(all_docs, embedding)
store = InMemoryStore()

# ParentDocumentRetriever 생성
pdr = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

print("ParentDocumentRetriever에 문서 추가를 시작합니다...")
pdr.add_documents(all_docs)
print("문서 추가 완료.")


# 3. 앙상블 검색을 위한 BM25 리트리버

parent_doc_ids = list(store.yield_keys())
parent_documents = store.mget(parent_doc_ids)

bm25_retriever = BM25Retriever.from_documents(documents=parent_documents)
bm25_retriever.k = 5


# --- 4. 필요한 객체들 저장 ---

# FAISS 벡터 DB 저장
vectorstore.save_local("advanced_rag_faiss_db")

# 부모 문서 저장소(docstore) 저장
with open("advanced_rag_docstore.pkl", "wb") as f:
    pickle.dump(store, f)

# BM25 리트리버를 위한 원본 부모 문서들 저장
with open("advanced_rag_parent_docs.pkl", "wb") as f:
    pickle.dump(parent_documents, f)

print("모든 인덱싱과 저장이 완료되었습니다.")
print("생성된 파일: 'advanced_rag_faiss_db' 폴더, 'advanced_rag_docstore.pkl', 'advanced_rag_parent_docs.pkl'")