import torch
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever, ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_transformers import LongContextReorder
import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.chat_message_histories import ChatMessageHistory

# --- 1. LLM 및 임베딩 모델 로딩 ---
embedding = HuggingFaceEmbeddings(model_name="jhgan/ko-sbert-nli")
quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
model_id = "naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id, device_map="auto", quantization_config=quantization_config, trust_remote_code=True
)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)
llm = HuggingFacePipeline(pipeline=pipe)

# --- 2. 작물별 리트리버 생성 함수 ---
def create_retriever_for_crop(crop_name_en: str):
    print(f"'{crop_name_en}' 작물의 DB를 로딩하여 리트리버를 생성합니다.")
    vectorstore = FAISS.load_local(f"{crop_name_en}_faiss_db", embedding, allow_dangerous_deserialization=True)
    with open(f"{crop_name_en}_parent_docs.pkl", "rb") as f: parent_documents = pickle.load(f)
    
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
    pdr = ParentDocumentRetriever(vectorstore=vectorstore, docstore=InMemoryStore(), child_splitter=child_splitter)
    
    bm25_retriever = BM25Retriever.from_documents(documents=parent_documents, k=5)
    ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, pdr], weights=[0.6, 0.4])
    return ensemble_retriever

# --- 3. 각 작물별 리트리버 미리 생성 ---
retrievers = {
    "딸기": create_retriever_for_crop("strawberry"),
    "감자": create_retriever_for_crop("potato"),
}
reordering = LongContextReorder()

# --- 4. 최종 답변 생성을 위한 프롬프트 (강화 버전) ---
# ❗️❗️❗️ 할루시네이션 방지를 위해 프롬프트를 대폭 수정 ❗️❗️❗️
answer_prompt = PromptTemplate.from_template(
    "당신은 '농업기술길잡이' 문서를 기반으로 답변하는 AI 전문가입니다.\n"
    "주어진 '내용'만을 사용하여 사용자의 '질문'에 대해 답변하세요.\n"
    "내용에 없는 정보는 절대로 언급하지 마세요. 추측하거나 꾸며내지 마세요.\n"
    "답변은 완전한 문장 형태로, 친절하고 상세하게 설명해주세요.\n\n"
    "--- 내용 ---\n{context}\n\n"
    "--- 질문 ---\n{question}\n\n"
    "--- 답변 ---\n"
)
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# --- 5. 메인 실행 로직 ---
# (쿼리 재작성 기능은 사용하지 않음)
print("안녕하세요! 농업 기술 챗봇입니다. '종료'를 입력하시면 대화가 끝납니다.")

while True:
    user_input = input("\n💬 질문: ")
    if user_input.lower() == "종료":
        print("대화를 종료합니다.")
        break

    question_for_rag = user_input
    print(f"💡 입력된 질문: {question_for_rag}")

    # 규칙 기반 라우팅
    is_strawberry = "딸기" in question_for_rag or "설향" in question_for_rag or "반촉성재배" in question_for_rag
    is_potato = "감자" in question_for_rag

    retrieved_docs = []
    if is_strawberry and is_potato:
        print(">> '딸기'와 '감자' DB에서 모두 정보를 검색합니다 (비교 질문).")
        strawberry_docs = retrievers["딸기"].invoke(question_for_rag)
        potato_docs = retrievers["감자"].invoke(question_for_rag)
        retrieved_docs = strawberry_docs + potato_docs
    elif is_strawberry:
        print(">> '딸기' DB에서 정보를 검색합니다.")
        retrieved_docs = retrievers["딸기"].invoke(question_for_rag)
    elif is_potato:
        print(">> '감자' DB에서 정보를 검색합니다.")
        retrieved_docs = retrievers["감자"].invoke(question_for_rag)

    if retrieved_docs:
        # 검색된 문서 순서 재배열 및 포맷팅
        reordered_docs = reordering.transform_documents(retrieved_docs)
        context_text = format_docs(reordered_docs)

        # 답변 생성 체인
        rag_chain = answer_prompt | llm | StrOutputParser()
        full_answer_output = rag_chain.invoke({
            "context": context_text,
            "question": question_for_rag
        })
        
        # ❗️❗️❗️ 프롬프트 전체 출력 문제를 해결하기 위한 파싱 로직 ❗️❗️❗️
        # LLM의 출력물에서 '--- 답변 ---' 이후의 내용만 깔끔하게 추출합니다.
        try:
            # 모델이 프롬프트를 그대로 출력하는 경우
            final_answer = full_answer_output.split("--- 답변 ---")[-1].strip()
            # 만약 답변이 비어있다면, 모델이 프롬프트를 출력하지 않고 바로 답변한 경우이므로 원본을 사용
            if not final_answer:
                final_answer = full_answer_output.strip()
        except:
            # 예외 발생 시 원본 출력 사용
            final_answer = full_answer_output.strip()

    else:
        print(">> 관련된 작물 정보를 찾을 수 없습니다.")
        final_answer = "죄송하지만, 해당 질문에 대한 정보는 제가 가진 문서에 없습니다. '딸기'나 '감자'에 대해 질문해주세요."

    print("\n✅ 답변:")
    print(final_answer)