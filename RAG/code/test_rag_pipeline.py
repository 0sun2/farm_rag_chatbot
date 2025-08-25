from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 임베딩 모델 로딩
embedding = HuggingFaceEmbeddings(model_name="jhgan/ko-sbert-nli")

# 2. 벡터 DB 불러오기
retriever = FAISS.load_local("strawberry_potato_FAISS_DB", embedding, allow_dangerous_deserialization=True).as_retriever()

# 3. 문서 분석 함수 추가
def analyze_documents(retriever):
    """문서의 길이와 복잡성을 분석하는 함수"""
    print("📊 문서 분석 시작...")
    
    # FAISS의 모든 문서 가져오기
    doc_dict = retriever.vectorstore.docstore._dict
    all_docs = list(doc_dict.values())
    
    doc_lengths = []
    doc_sources = []
    doc_complexity = []
    
    for doc in all_docs:
        content = doc.page_content
        length = len(content)
        doc_lengths.append(length)
        
        # 출처 정보 추출
        source = doc.metadata.get('source', 'unknown')
        doc_sources.append(source)
        
        # 복잡성 지표 (문장 수, 단어 수 등)
        sentences = content.count('.') + content.count('!') + content.count('?')
        words = len(content.split())
        complexity = words / max(sentences, 1)  # 문장당 평균 단어 수
        doc_complexity.append(complexity)
    
    # 분석 결과 출력
    print(f"\n📈 문서 통계:")
    print(f"총 문서 수: {len(doc_lengths)}")
    print(f"평균 길이: {sum(doc_lengths)/len(doc_lengths):.1f}자")
    print(f"최대 길이: {max(doc_lengths)}자")
    print(f"최소 길이: {min(doc_lengths)}자")
    print(f"평균 복잡성 (문장당 단어 수): {sum(doc_complexity)/len(doc_complexity):.1f}")
    
    # 길이별 분포
    print(f"\n📏 길이별 분포:")
    length_ranges = {
        "매우 짧음 (0-100자)": len([l for l in doc_lengths if l <= 100]),
        "짧음 (101-300자)": len([l for l in doc_lengths if 101 <= l <= 300]),
        "보통 (301-500자)": len([l for l in doc_lengths if 301 <= l <= 500]),
        "길음 (501-800자)": len([l for l in doc_lengths if 501 <= l <= 800]),
        "매우 길음 (800자+)": len([l for l in doc_lengths if l > 800])
    }
    
    for range_name, count in length_ranges.items():
        percentage = (count / len(doc_lengths)) * 100
        print(f"  {range_name}: {count}개 ({percentage:.1f}%)")
    
    # 출처별 분석
    print(f"\n📁 출처별 분석:")
    source_counts = {}
    for source in doc_sources:
        source_counts[source] = source_counts.get(source, 0) + 1
    
    for source, count in source_counts.items():
        print(f"  {source}: {count}개 문서")
    
    # 복잡성 분석
    print(f"\n🧠 복잡성 분석:")
    complexity_ranges = {
        "단순 (문장당 10단어 이하)": len([c for c in doc_complexity if c <= 10]),
        "보통 (문장당 11-20단어)": len([c for c in doc_complexity if 11 <= c <= 20]),
        "복잡 (문장당 21단어 이상)": len([c for c in doc_complexity if c > 20])
    }
    
    for complexity_name, count in complexity_ranges.items():
        percentage = (count / len(doc_complexity)) * 100
        print(f"  {complexity_name}: {count}개 ({percentage:.1f}%)")
    
    # RAG 기법 추천
    print(f"\n💡 RAG 기법 추천:")
    
    avg_length = sum(doc_lengths)/len(doc_lengths)
    avg_complexity = sum(doc_complexity)/len(doc_complexity)
    
    if avg_length > 500:
        print("  ✅ ParentDocumentRetriever 추천: 문서가 길어서 문맥 보존이 중요")
    elif avg_length < 200:
        print("  ✅ 기본 Retriever로 충분: 문서가 짧아서 추가 분할 불필요")
    
    if avg_complexity > 15:
        print("  ✅ Multi-Query Retriever 고려: 복잡한 문장 구조로 인한 검색 어려움 가능")
    
    if len(set(doc_sources)) > 3:
        print("  ✅ Self-Querying 고려: 다양한 출처의 메타데이터 활용 가능")
    
    return {
        'lengths': doc_lengths,
        'sources': doc_sources,
        'complexity': doc_complexity,
        'avg_length': avg_length,
        'avg_complexity': avg_complexity
    }

# 4. 문서 분석 실행
print("🔍 문서 특성 분석 중...")
analysis_result = analyze_documents(retriever)

# 5. Llama3 모델 로딩
model_id = "naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B"
tokenizer = AutoTokenizer.from_pretrained(model_id,trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", load_in_4bit=True,trust_remote_code=True)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=384,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.1
)

llm = HuggingFacePipeline(pipeline=pipe)

# 6. 사용자 질문 프롬프트 정의
prompt_template = """
당신은 농업 분야의 전문가입니다. 아래 문서 내용을 참고해 사용자의 질문에 정확하고 간결하게 답변해주세요.
모르는 내용은 '잘 모르겠습니다.'라고 말해주세요.

문서:
{context}

질문: {question}
답변:
"""
prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)

# 7. QA 체인 구성
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)

# 8. 질문 실행
query = "딸기를 심으려면 어떻게 해야돼?"
result = qa_chain.invoke(query)

print("\n💬 답변:", result["result"])
print("\n📄 참고 문서:", [doc.metadata.get("source", "출처 없음") for doc in result["source_documents"]])

# 9. 검색된 문서의 길이 확인
print(f"\n🔍 검색된 문서 분석:")
for i, doc in enumerate(result["source_documents"]):
    doc_length = len(doc.page_content)
    print(f"문서 {i+1}: {doc_length}자 - {doc.metadata.get('source', '출처 없음')}")
    
    # 문서 내용 미리보기
    preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
    print(f"  미리보기: {preview}")
    print()
