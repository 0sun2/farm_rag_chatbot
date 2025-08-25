import streamlit as st

# 페이지 설정
st.set_page_config(
    page_title="🌱 테스트",
    page_icon="🌱",
    layout="wide"
)

# 메인 헤더
st.markdown("## 🌱 농업 기술 챗봇 테스트")
st.markdown("### 기본 Streamlit이 작동하는지 확인합니다.")

# 간단한 입력
user_input = st.text_input("테스트 입력:", "안녕하세요!")

if st.button("확인"):
    st.success(f"입력된 내용: {user_input}")

# 파일 존재 여부 확인
import os

st.markdown("### 📁 파일 존재 여부 확인")

files_to_check = [
    "strawberry_faiss_db",
    "strawberry_parent_docs.pkl", 
    "potato_faiss_db",
    "potato_parent_docs.pkl"
]

for file_path in files_to_check:
    if os.path.exists(file_path):
        st.success(f"✅ {file_path} - 존재함")
    else:
        st.error(f"❌ {file_path} - 없음")

# 패키지 버전 확인
st.markdown("### 📦 패키지 버전 확인")

try:
    import langchain
    st.info(f"LangChain: {langchain.__version__}")
except Exception as e:
    st.error(f"LangChain 로딩 실패: {e}")

try:
    import transformers
    st.info(f"Transformers: {transformers.__version__}")
except Exception as e:
    st.error(f"Transformers 로딩 실패: {e}")

try:
    import torch
    st.info(f"PyTorch: {torch.__version__}")
except Exception as e:
    st.error(f"PyTorch 로딩 실패: {e}") 