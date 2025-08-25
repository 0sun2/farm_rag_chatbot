from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain.document_transformers import LongContextReorder
import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- 1. LLM 및 임베딩 모델 로딩 ---
embedding = HuggingFaceEmbeddings(model_name="jhgan/ko-sbert-nli")

model_id = "naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", load_in_4bit=True, trust_remote_code=True)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.1
)
llm = HuggingFacePipeline(pipeline=pipe)

# --- 2. 저장된 DB 및 문서 저장소 불러오기 ---
print("저장된 DB와 문서를 불러옵니다...")
vectorstore = FAISS.load_local("advanced_rag_faiss_db", embedding, allow_dangerous_deserialization=True)
with open("advanced_rag_docstore.pkl", "rb") as f:
    store = pickle.load(f)
with open("advanced_rag_parent_docs.pkl", "rb") as f:
    parent_documents = pickle.load(f)
print("로딩 완료.")

# --- 3. 고급 리트리버 구성 ---
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
pdr = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
)
pdr.search_kwargs = {'k': 5}
bm25_retriever = BM25Retriever.from_documents(documents=parent_documents)
bm25_retriever.k = 5
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, pdr],
    weights=[0.6, 0.4]
)

# --- 4. LongContextReorder 후처리 설정 ---
reordering = LongContextReorder()

# --- 5. LCEL을 사용한 체인 구성 ---
prompt_template = """
주어진 내용을 바탕으로 다음 질문에 답변해 주세요.

내용:
{context}

질문: {question}

답변:
"""
prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# ❗️❗️❗️ 에러 해결 부분 ❗️❗️❗️
# reordering(DocumentTransformer)을 체인에 통합하기 위해 함수로 감싸줍니다.
# retriever가 문서를 반환하면(docs), reordering.transform_documents(docs)로 순서를 바꾸고
# 그 결과를 format_docs 함수로 넘겨 하나의 문자열로 만듭니다.
# RunnableLambda는 이 함수를 LCEL 체인에서 사용할 수 있게 해줍니다.
reorder_and_format = RunnableLambda(
    lambda docs: format_docs(reordering.transform_documents(docs))
)


# LCEL 체인 구성
# 이제 retriever의 출력을 바로 reorder_and_format 람다 함수에 연결합니다.
rag_chain = (
    {"context": ensemble_retriever | reorder_and_format, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# --- 6. 질문 실행 ---
query = "딸기 촉성재배를 할 때 온도 관리는 어떻게 해야돼? 생육 단계별로 상세히 알려줘."
print("\n💬 질문:", query)
result = rag_chain.invoke(query)
print("\n✅ 답변:")
print(result)

print("\n\n📄 참고 문서 확인:")
retrieved_docs = ensemble_retriever.invoke(query)
for i, doc in enumerate(retrieved_docs):
    print(f"--- 문서 {i+1} (출처: {doc.metadata.get('source', '출처 없음')}, 페이지: {doc.metadata.get('page', 'N/A')}) ---")
    print(doc.page_content[:300] + "...")
    print("-" * 50)