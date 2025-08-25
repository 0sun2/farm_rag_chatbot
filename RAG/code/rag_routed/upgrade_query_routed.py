import torch
# --- DeprecationWarning 해결을 위해 import 경로를 최신으로 수정합니다 ---
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings # 수정
from langchain_community.retrievers import BM25Retriever # 수정
from langchain.retrievers import EnsembleRetriever, ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableBranch
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_transformers import LongContextReorder # 수정
import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.docstore.in_memory import InMemoryDocstore # 수정

# --- 1. LLM 및 임베딩 모델 로딩 ---
embedding = HuggingFaceEmbeddings(model_name="jhgan/ko-sbert-nli")
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)
model_id = "naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    device_map="auto", 
    quantization_config=quantization_config, # 최신 방식 적용
    trust_remote_code=True,
)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)
llm = HuggingFacePipeline(pipeline=pipe)

# --- 2. 작물별 RAG 체인 생성 함수 ---
def create_rag_chain_for_crop(crop_name_en: str):
    print(f"'{crop_name_en}' 작물의 DB를 로딩하여 체인을 생성합니다.")
    vectorstore = FAISS.load_local(f"{crop_name_en}_faiss_db", embedding, allow_dangerous_deserialization=True)
    with open(f"{crop_name_en}_docstore.pkl", "rb") as f: store = pickle.load(f)
    with open(f"{crop_name_en}_parent_docs.pkl", "rb") as f: parent_documents = pickle.load(f)

    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
    pdr = ParentDocumentRetriever(vectorstore=vectorstore, docstore=store, child_splitter=child_splitter)
    bm25_retriever = BM25Retriever.from_documents(documents=parent_documents, k=5)
    ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, pdr], weights=[0.6, 0.4])
    reordering = LongContextReorder()
    prompt = PromptTemplate.from_template("주어진 내용을 바탕으로 다음 질문에 답변해 주세요.\n\n내용:\n{context}\n\n질문: {question}\n\n답변:")

    def format_docs(docs): return "\n\n".join(doc.page_content for doc in docs)
    reorder_and_format = RunnableLambda(lambda docs: format_docs(reordering.transform_documents(docs)))
    rag_chain = ({"context": ensemble_retriever | reorder_and_format, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser())
    return rag_chain

# --- 3. 라우터(Router) 체인 구성 ---
router_template = """당신은 사용자의 질문을 '딸기', '감자', '일반' 세 가지 카테고리 중 하나로 분류하는 라우터입니다. 질문에 어떤 작물 이름이 명시적으로 언급되는지 확인하고, 해당하는 카테고리 이름만 답변하세요. 질문: {question} 카테고리:"""
router_prompt = ChatPromptTemplate.from_template(router_template)
router_chain = router_prompt | llm | StrOutputParser()

# --- 4. 각 작물별 체인과 일반 체인 미리 생성 ---
rag_chains = {
    "딸기": create_rag_chain_for_crop("strawberry"),
    "감자": create_rag_chain_for_crop("potato"),
}
general_chain = RunnableLambda(lambda x: "어떤 작물에 대해 질문하시는지 명확히 말씀해주세요. (예: 딸기, 감자)")

# --- 5. RunnableBranch를 이용한 최종 체인 결합 ---
# ❗️❗️❗️ 에러 해결 부분 ❗️❗️❗️
# 라우터의 출력(topic)과 질문(question)을 모두 받아서 처리하는 함수를 정의합니다.
def route(info):
    topic_output = info["topic"].strip().lower() # LLM의 출력을 소문자로 바꾸고 공백 제거
    question = info["question"]
    
    # 정확한 일치 대신, 키워드 포함 여부로 분기합니다.
    if "딸기" in topic_output:
        print(">> '딸기' 체인으로 라우팅합니다.")
        return rag_chains["딸기"].invoke(question) # 해당 체인에 질문을 직접 전달하여 실행
    elif "감자" in topic_output:
        print(">> '감자' 체인으로 라우팅합니다.")
        return rag_chains["감자"].invoke(question)
    else:
        print(">> '일반' 체인으로 라우팅합니다.")
        return general_chain.invoke(question)

# 전체 파이프라인: router의 결과와 원본 질문을 함께 route 함수로 전달
full_chain = {"topic": router_chain, "question": lambda x: x["question"]} | RunnableLambda(route)

# --- 6. 질문 실행 ---
def ask_question(query: str):
    print(f"\n💬 질문: {query}")
    result = full_chain.invoke({"question": query})
    print("\n✅ 답변:")
    print(result)
    print("\n" + "="*50)

# 테스트 질문
ask_question("딸기 촉성재배 시 온도 관리는 어떻게 하나요?")
ask_question("감자 시설재배에서 가장 중요한 환경요인은 뭐야?")
ask_question("배추는 어떻게 키워?")