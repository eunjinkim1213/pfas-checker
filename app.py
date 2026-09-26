import streamlit as st
import google.generativeai as genai
import pandas as pd
from PIL import Image

# 1. 화면 제목 및 안내
st.set_page_config(page_title="EU PFAS Checker", page_icon="🧪")
st.title("🧪 EU REACH·PFAS 화학성분 판별기")
st.write("화학제품 라벨이나 성분표를 사진으로 찍어 올리면, AI가 유럽 규제 물질을 찾아냅니다.")

# 2. 무료 API 키 입력 (보안)
api_key = st.text_input("구글 AI API 키를 입력하세요:", type="password")

# 3. 규제 DB 불러오기
try:
    db = pd.read_csv("pfas_list.csv")
except:
    st.error("pfas_list.csv 파일이 없습니다.")

# 4. 스마트폰 카메라 촬영 또는 사진 업로드
uploaded_file = st.camera_input("라벨 사진을 촬영하세요") or st.file_uploader("또는 사진 파일을 올리세요", type=["jpg", "png", "jpeg"])

if uploaded_file and api_key:
    # 이미지 화면 표시
    image = Image.open(uploaded_file)
    st.image(image, caption="분석할 라벨 사진", use_container_width=True)
    
    with st.spinner("AI가 라벨을 읽고 유럽 규제 DB와 대조 중입니다..."):
        try:
            # 구글 Gemini AI 세팅
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-3.8-flash") # 완전 무료 모델
            
            # AI에게 시킬 일 (프롬프트)
            prompt = f"""
            너는 유럽 화학물질 규제(EU REACH 및 PFAS) 전문가야.
            사진 속 화학제품 라벨에서 '화학물질 이름'이나 'CAS 번호'를 모두 읽어줘.
            
            그리고 우리가 가진 다음 규제 목록과 대조해줘:
            {db.to_string()}
            
            다음 형식으로만 깔끔하게 답변해줘:
            1. 발견된 주요 성분 및 CAS 번호
            2. 규제 상태 판정 (🟢 안전 / 🟡 주의 / 🔴 위험-PFAS규제대상)
            3. 화공/반도체 공정 관점의 한 줄 설명 (예: 반도체 세정제용 대체 물질 필요 여부 등)
            """
            
            response = model.generate_content([prompt, image])
            
            # 결과 출력
            st.success("분석 완료!")
            st.markdown(response.text)
            
        except Exception as e:
            st.error(f"오류가 발생했습니다: {e}")
