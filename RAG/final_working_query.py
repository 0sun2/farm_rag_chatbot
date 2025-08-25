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

# --- 4. 쿼리 재작성 체인 (제거됨) ---
# 이 부분의 코드가 제거되었습니다.

# --- 5. 최종 답변 생성 체인 ---
answer_prompt = PromptTemplate.from_template("주어진 내용을 바탕으로 다음 질문에 상세하고 친절하게 답변해 주세요.\n\n내용:\n{context}\n\n질문: {input}\n\n답변:")
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# --- 6. 메인 실행 로직 ---
chat_history = ChatMessageHistory()
print("안녕하세요! 농업 기술 챗봇입니다. '종료'를 입력하시면 대화가 끝납니다.")

while True:
    user_input = input("\n💬 질문: ")
    if user_input.lower() == "종료":
        print("대화를 종료합니다.")
        break

    # --- 쿼리 재작성 단계가 제거됨 ---
    # 이제 사용자의 입력을 바로 라우팅과 검색에 사용합니다.
    question_for_rag = user_input
    print(f"💡 입력된 질문: {question_for_rag}")

    # 규칙 기반 라우팅
    is_strawberry = "딸기" in question_for_rag or "설향" in question_for_rag
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

        # 답변 생성
        rag_chain = answer_prompt | llm | StrOutputParser()
        answer = rag_chain.invoke({
            "context": context_text,
            "input": question_for_rag
        })
    else:
        print(">> 관련된 작물 정보를 찾을 수 없습니다.")
        answer = "죄송하지만, 해당 질문에 대한 정보는 제가 가진 문서에 없습니다. '딸기'나 '감자'에 대해 질문해주세요."

    print("\n✅ 답변:")
    print(answer)
    
    # 대화 기록은 여전히 저장하여 대화의 흐름을 볼 수 있습니다.
    chat_history.add_user_message(user_input)
    chat_history.add_ai_message(answer)