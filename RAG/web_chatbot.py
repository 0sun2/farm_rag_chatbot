import streamlit as st
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

# 페이지 설정
st.set_page_config(
    page_title="🌱 농업 기술 챗봇",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일링
st.markdown("""
<style>
    /* 전체 컨테이너의 최대 가로폭 제한 */
    .block-container {
        max-width: 950px !important;
        margin-left: auto !important;
        margin-right: auto !important;
    }
    /* 전체 페이지 배율(zoom) 조정 */
    html {
        zoom: 1.12;
    }
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #2E8B57;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #E8F5E8;
        border-left: 5px solid #2E8B57;
    }
    .bot-message {
        background-color: #F0F8FF;
        border-left: 5px solid #4682B4;
    }

    /* 🎨 답변 가독성 향상을 위한 스타일 추가 */
    .bot-message p {
        line-height: 1.7; /* 문장 줄간격 */
        margin-bottom: 1rem; /* 단락 아래 여백 */
    }
    .bot-message ul, .bot-message ol {
        padding-left: 1.5rem; /* 목록 들여쓰기 */
        margin-bottom: 1rem; /* 목록 전체 아래 여백 */
    }
    .bot-message li {
        margin-bottom: 0.75rem; /* 목록 항목 간 여백 */
        line-height: 1.7; /* 목록 항목 내 줄간격 */
    }
    /* 마지막 p, ul, ol 요소의 아래 여백 제거하여 컨테이너와 간격 일관성 유지 */
    .bot-message p:last-child,
    .bot-message ul:last-child,
    .bot-message ol:last-child {
        margin-bottom: 0;
    }
    /* --- 기존 스타일 --- */
    
    .stButton > button {
        background-color: #2E8B57;
        color: white;
        border-radius: 0.5rem;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #228B22;
    }
    .sidebar .sidebar-content {
        background-color: #F8F9FA;
    }
    .info-box {
        background-color: #E3F2FD;
        border-left: 5px solid #2196F3;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# 사이드바
with st.sidebar:
    st.markdown("## 🌱 농업 기술 챗봇")
    st.markdown("---")
    st.markdown("### 📚 지원 작물")
    st.markdown("- 🍓 딸기")
    st.markdown("- 🥔 감자")
    st.markdown("---")
    st.markdown("### 💡 질문 예시")
    st.markdown("- 딸기를 심으려면 어떻게 해야돼?")
    st.markdown("- 감자 재배 시 주의사항은?")
    st.markdown("- 딸기와 감자 중 어떤 게 더 쉬워?")
    st.markdown("- 딸기 반촉성재배 방법은?")
    st.markdown("---")
    st.markdown("### 🔧 사용된 기술")
    st.markdown("- **RAG (Retrieval-Augmented Generation)**")
    st.markdown("- **ParentDocumentRetriever**")
    st.markdown("- **Ensemble Retriever**")
    st.markdown("- **Long Context Reorder**")

# 메인 헤더
st.markdown('<h1 class="main-header">🌱 농업 기술 챗봇</h1>', unsafe_allow_html=True)
st.markdown("### 딸기와 감자 재배에 대한 모든 궁금증을 해결해드립니다 🚀")

# 모델 로딩 (캐싱)
@st.cache_resource
def load_models():
    """모델들을 로딩하고 캐싱하는 함수"""
    with st.spinner("🤖 AI 모델을 로딩하고 있습니다..."):
        # 임베딩 모델
        embedding = HuggingFaceEmbeddings(model_name="jhgan/ko-sbert-nli")
        
        # LLM 모델
        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        model_id = "naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B"
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="auto", quantization_config=quantization_config, trust_remote_code=True
        )
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)
        llm = HuggingFacePipeline(pipeline=pipe)
        
        return embedding, llm

@st.cache_resource
def create_retrievers(_embedding):
    """리트리버들을 생성하고 캐싱하는 함수"""
    with st.spinner("📚 데이터베이스를 로딩하고 있습니다..."):
        def create_retriever_for_crop(crop_name_en: str):
            try:
                # FAISS 벡터 데이터베이스 로딩
                vectorstore = FAISS.load_local(f"{crop_name_en}_faiss_db", _embedding, allow_dangerous_deserialization=True)
                
                # 부모 문서 로딩
                with open(f"{crop_name_en}_parent_docs.pkl", "rb") as f: 
                    parent_documents = pickle.load(f)
                
                # ParentDocumentRetriever 설정
                child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
                pdr = ParentDocumentRetriever(vectorstore=vectorstore, docstore=InMemoryStore(), child_splitter=child_splitter)
                
                # BM25 리트리버 설정
                bm25_retriever = BM25Retriever.from_documents(documents=parent_documents, k=5)
                
                # Ensemble 리트리버 생성
                ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, pdr], weights=[0.6, 0.4])
                return ensemble_retriever
            except Exception as e:
                st.error(f"❌ {crop_name_en} 데이터베이스 로딩 실패: {str(e)}")
                st.error(f"📁 확인할 파일들:")
                st.error(f"   - {crop_name_en}_faiss_db/ (폴더)")
                st.error(f"   - {crop_name_en}_parent_docs.pkl")
                return None
        
        retrievers = {
            "딸기": create_retriever_for_crop("strawberry"),
            "감자": create_retriever_for_crop("potato"),
        }
        
        return retrievers

# 모델 로딩
embedding, llm = load_models()
retrievers = create_retrievers(embedding)
reordering = LongContextReorder()

# 답변 생성 함수
def generate_answer(question, retrieved_docs):
    """답변을 생성하는 함수"""
    if not retrieved_docs:
        return "죄송하지만, 해당 질문에 대한 정보는 제가 가진 문서에 없습니다. '딸기'나 '감자'에 대해 질문해주세요."
    
    # 검색된 문서 순서 재배열 및 포맷팅
    reordered_docs = reordering.transform_documents(retrieved_docs)
    context_text = "\n\n".join(doc.page_content for doc in reordered_docs)
    
    # 강화된 프롬프트 (할루시네이션 방지)
    answer_prompt = PromptTemplate.from_template(
        "당신은 '농업기술길잡이' 문서를 기반으로 답변하는 AI 전문가입니다.\n"
        "주어진 '내용'만을 사용하여 사용자의 '질문'에 대해 답변하세요.\n"
        "내용에 없는 정보는 절대로 언급하지 마세요. 추측하거나 꾸며내지 마세요.\n"
        "답변은 완전한 문장 형태로, 친절하고 상세하게 설명해주세요.\n\n"
        "--- 내용 ---\n{context}\n\n"
        "--- 질문 ---\n{question}\n\n"
        "--- 답변 ---\n"
    )
    
    rag_chain = answer_prompt | llm | StrOutputParser()
    
    with st.spinner("🤔 답변을 생성하고 있습니다..."):
        full_answer_output = rag_chain.invoke({
            "context": context_text,
            "question": question
        })
        
        # 프롬프트 전체 출력 문제를 해결하기 위한 파싱 로직
        try:
            # 모델이 프롬프트를 그대로 출력하는 경우
            final_answer = full_answer_output.split("--- 답변 ---")[-1].strip()
            # 만약 답변이 비어있다면, 모델이 프롬프트를 출력하지 않고 바로 답변한 경우이므로 원본을 사용
            if not final_answer:
                final_answer = full_answer_output.strip()
        except:
            # 예외 발생 시 원본 출력 사용
            final_answer = full_answer_output.strip()
    
    return final_answer

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = ChatMessageHistory()

# 채팅 메시지 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력
if prompt := st.chat_input("딸기나 감자에 대해 궁금한 점을 물어보세요!"):
    # 사용자 메시지 추가
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # 챗봇 응답 생성
    with st.chat_message("assistant"):
        with st.spinner("🔍 관련 정보를 검색하고 있습니다..."):
            # 규칙 기반 라우팅 (원본 파일과 동일)
            is_strawberry = "딸기" in prompt or "설향" in prompt or "반촉성재배" in prompt
            is_potato = "감자" in prompt
            
            retrieved_docs = []
            search_info = ""
            
            if is_strawberry and is_potato:
                search_info = "🍓 '딸기'와 🥔 '감자' DB에서 모두 정보를 검색합니다 (비교 질문)."
                strawberry_docs = retrievers["딸기"].invoke(prompt) if retrievers["딸기"] else []
                potato_docs = retrievers["감자"].invoke(prompt) if retrievers["감자"] else []
                retrieved_docs = strawberry_docs + potato_docs
            elif is_strawberry:
                search_info = "🍓 '딸기' DB에서 정보를 검색합니다."
                retrieved_docs = retrievers["딸기"].invoke(prompt) if retrievers["딸기"] else []
            elif is_potato:
                search_info = "🥔 '감자' DB에서 정보를 검색합니다."
                retrieved_docs = retrievers["감자"].invoke(prompt) if retrievers["감자"] else []
            
            if search_info:
                st.info(search_info)
            
            # 답변 생성
            response = generate_answer(prompt, retrieved_docs)
            st.markdown(response)
    
    # 챗봇 메시지 추가
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    # 대화 기록 저장
    st.session_state.chat_history.add_user_message(prompt)
    st.session_state.chat_history.add_ai_message(response)

# 하단 정보
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("### 📊 사용된 기술")
    st.markdown("- RAG (Retrieval-Augmented Generation)")
    st.markdown("- ParentDocumentRetriever")
    st.markdown("- Ensemble Retriever")

with col2:
    st.markdown("### 🌱 지원 작물")
    st.markdown("- 🍓 딸기 재배법")
    st.markdown("- 🥔 감자 재배법")
    st.markdown("- 📚 농업 기술 가이드")

with col3:
    st.markdown("### 💡 사용 팁")
    st.markdown("- 구체적인 질문을 해주세요")
    st.markdown("- 딸기/감자 키워드를 포함하세요")
    st.markdown("- 비교 질문도 가능합니다")

# 대화 초기화 버튼
if st.button("🗑️ 대화 초기화"):
    st.session_state.messages = []
    st.session_state.chat_history = ChatMessageHistory()
    st.rerun() 