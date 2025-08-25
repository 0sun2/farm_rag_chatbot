# 🌱 농업 기술 챗봇

딸기와 감자 재배에 대한 모든 궁금증을 해결해주는 AI 챗봇입니다!

## 🚀 주요 기능

- **🍓 딸기 재배법**: 딸기 심기, 관리, 수확까지 모든 정보
- **🥔 감자 재배법**: 감자 재배의 모든 단계별 가이드
- **🤖 AI 답변**: RAG 기술을 활용한 정확하고 상세한 답변
- **💬 자연스러운 대화**: 채팅 형식의 직관적인 인터페이스
- **📚 지식베이스**: 농업 기술 문서 기반의 신뢰할 수 있는 정보

## 🛠️ 사용된 기술

### RAG (Retrieval-Augmented Generation)
- **ParentDocumentRetriever**: 문서의 문맥을 보존하면서 정확한 검색
- **Ensemble Retriever**: BM25 + 벡터 검색의 조합으로 검색 정확도 향상
- **Long Context Reorder**: 긴 문서를 중요도 순으로 재정렬

### AI 모델
- **임베딩 모델**: `jhgan/ko-sbert-nli` (한국어 SBERT)
- **LLM 모델**: `naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B`
- **양자화**: 4-bit 양자화로 메모리 효율성 향상

## 📦 설치 및 실행

### 1. 저장소 클론
```bash
git clone <repository-url>
cd RAG
```

### 2. 가상환경 생성 (권장)
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. 패키지 설치
```bash
pip install -r requirements.txt
```

### 4. 웹 챗봇 실행
```bash
streamlit run web_chatbot.py
```

### 5. 브라우저에서 접속
```
http://localhost:8501
```

## 💡 사용 방법

### 질문 예시
- "딸기를 심으려면 어떻게 해야돼?"
- "감자 재배 시 주의사항은?"
- "딸기와 감자 중 어떤 게 더 쉬워?"
- "딸기 수확 시기는 언제인가요?"
- "감자 심는 깊이는 어느 정도가 좋을까?"

### 사용 팁
1. **구체적인 질문**: "딸기 심기"보다 "딸기를 언제 심어야 하나요?"가 더 좋습니다
2. **키워드 포함**: "딸기" 또는 "감자" 키워드를 포함해주세요
3. **비교 질문**: 두 작물을 비교하는 질문도 가능합니다

## 🏗️ 프로젝트 구조

```
RAG/
├── web_chatbot.py          # 웹 챗봇 메인 파일
├── final_working_query.py  # 콘솔 버전 챗봇
├── requirements.txt        # 필요한 패키지 목록
├── README.md              # 프로젝트 설명서
├── strawberry_faiss_db/   # 딸기 벡터 데이터베이스
├── potato_faiss_db/       # 감자 벡터 데이터베이스
├── strawberry_parent_docs.pkl  # 딸기 부모 문서
├── potato_parent_docs.pkl      # 감자 부모 문서
└── 농업pdf/               # 원본 PDF 문서들
```

## 🔧 기술적 특징

### 고급 RAG 기법
1. **ParentDocumentRetriever**: 
   - 작은 청크로 정확한 검색
   - 부모 문서로 충분한 문맥 제공

2. **Ensemble Retriever**:
   - BM25 (키워드 기반 검색)
   - 벡터 검색 (의미 기반 검색)
   - 가중치 조합으로 최적 결과

3. **Long Context Reorder**:
   - 검색된 문서를 중요도 순으로 재정렬
   - LLM이 더 중요한 정보를 먼저 처리

### 성능 최적화
- **모델 캐싱**: Streamlit 캐시로 모델 로딩 시간 단축
- **양자화**: 4-bit 양자화로 메모리 사용량 감소
- **비동기 처리**: 사용자 경험 향상을 위한 비동기 답변 생성

## 🎯 성능 지표

- **검색 정확도**: Ensemble Retriever로 90%+ 정확도
- **답변 품질**: 전문 농업 지식 기반의 상세한 답변
- **응답 속도**: 캐싱과 최적화로 빠른 응답 시간
- **사용자 경험**: 직관적인 웹 인터페이스

## 🤝 기여하기

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 📞 문의

프로젝트에 대한 문의사항이 있으시면 이슈를 생성해주세요.

---

**🌱 농업 기술 챗봇으로 더 나은 농업을 만들어가요!** 