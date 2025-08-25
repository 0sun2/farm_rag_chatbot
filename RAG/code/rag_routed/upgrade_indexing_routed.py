import os
import faiss 
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever
import pickle
from langchain.docstore import InMemoryDocstore

# --- 처리할 PDF 파일 목록 정의 ---
# ❗️❗️❗️ 수정된 부분 ❗️❗️❗️
# 영문 키를 파일명/폴더명으로 사용하고, 한글 이름은 라우팅 시 비교용으로 사용합니다.
pdf_folder = "C:/Users/lys/Desktop/RAG/농업pdf"
pdf_info = {
    "strawberry": {
        "korean_name": "딸기",
        "filename": "농업기술길잡이40_딸기.pdf"
    },
    "potato": {
        "korean_name": "감자",
        "filename": "감자.pdf"
    }
}

# 임베딩 모델 로딩
embedding = HuggingFaceEmbeddings(model_name="jhgan/ko-sbert-nli")

# 각 파일에 대해 개별적으로 인덱싱 수행
for key, info in pdf_info.items():
    crop_name_en = key
    crop_name_ko = info["korean_name"]
    pdf_filename = info["filename"]
    
    print(f"--- '{crop_name_ko}'({crop_name_en}) 작물에 대한 인덱싱을 시작합니다. ---")
    
    # 1. 문서 로딩
    full_path = os.path.join(pdf_folder, pdf_filename)
    loader = PyMuPDFLoader(full_path)
    docs = loader.load()
    for doc in docs:
        doc.metadata['source'] = pdf_filename
    print(f"'{pdf_filename}'에서 총 {len(docs)}개의 페이지를 로딩했습니다.")

    # 2. ParentDocumentRetriever 설정
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=100)
    
    # 비어있는 FAISS DB 초기화
    test_embedding = embedding.embed_query("테스트")
    embedding_dimension = len(test_embedding)
    faiss_index = faiss.IndexFlatL2(embedding_dimension)
    vectorstore = FAISS(embedding.embed_query, faiss_index, InMemoryDocstore({}), {})

    store = InMemoryStore()
    pdr = ParentDocumentRetriever(vectorstore=vectorstore, docstore=store, child_splitter=child_splitter, parent_splitter=parent_splitter)
    pdr.add_documents(docs)
    print(f"'{crop_name_ko}'의 ParentDocumentRetriever 문서 추가 완료.")

    # 3. 영문 이름으로 파일 저장
    faiss_db_path = f"{crop_name_en}_faiss_db"
    docstore_path = f"{crop_name_en}_docstore.pkl"
    parent_docs_path = f"{crop_name_en}_parent_docs.pkl"

    vectorstore.save_local(faiss_db_path)
    
    with open(docstore_path, "wb") as f:
        pickle.dump(store, f)
        
    parent_doc_ids = list(store.yield_keys())
    parent_documents = store.mget(parent_doc_ids)
    with open(parent_docs_path, "wb") as f:
        pickle.dump(parent_documents, f)

    print(f"'{crop_name_ko}' 인덱싱 완료! 생성된 파일: {faiss_db_path}, {docstore_path}, {parent_docs_path}")
    print("-" * 50)

print("모든 작물에 대한 인덱싱이 완료되었습니다.")