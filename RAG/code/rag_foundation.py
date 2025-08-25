from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# 1. 임베딩 모델 로딩
embedding = HuggingFaceEmbeddings(model_name="jhgan/ko-sbert-nli")

# 2. 벡터 DB 불러오기
retriever = FAISS.load_local("strawberry_potato_FAISS_DB", embedding, allow_dangerous_deserialization=True).as_retriever()

# 3. Llama3 모델 로딩
model_id = "models--Qwen--Qwen3-8B"
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

# 4. 사용자 질문 프롬프트 정의
prompt_template = """
당신은 농업 분야의 전문가입니다. 아래 문서 내용을 참고해 사용자의 질문에 정확하고 간결하게 답변해주세요.
모르는 내용은 '잘 모르겠습니다.'라고 말해주세요.

문서:
{context}

질문: {question}
답변:
"""
prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)

# 5. QA 체인 구성
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)

# 6. 질문 실행
query = "딸기를 심으려면 어떻게 해야돼?"
result = qa_chain.invoke(query)

print("\n💬 답변:", result["result"])
print("\n📄 참고 문서:", [doc.metadata.get("source", "출처 없음") for doc in result["source_documents"]])
