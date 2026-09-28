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
너는 EU REACH, CLP 및 한국 화학물질 규제(PFAS 포함) 전문가야.
사진 속 라벨에서 성분명, CAS 번호, 위험 표시를 읽고 판정해줘.

우리가 가진 규제 DB 목록:
{db.to_string()}

[판정 가이드라인 - 엄격 적용]
1. 🔴 위험 (PFAS 규제대상 / 고위험):
   - 위 규제 DB에 매칭되는 물질이 있는 경우
   - 성분명에 "불소", "불소수지", "불소계", "PTFE", "Fluoro", "PFAS" 단어가 포함된 경우
   - 라벨에 부식성, 급성독성(해골), 환경유해성 픽토그램이 보이거나 강한 유해 화학물질(차아염소산나트륨, 수산화나트륨 등)인 경우
2. 🟡 주의:
   - 인화성(불꽃), 자극성(느낌표) 픽토그램이 있거나 에탄올, 유기용제 등이 주성분인 경우
3. 🟢 안전:
   - 위험 픽토그램이 전혀 없고, 무불소(PFC-Free) 또는 친환경/천연 계면활성제 등 안전 성분만 있는 경우

다음 형식으로만 깔끔하게 답변해줘:
1. 발견된 주요 성분 및 CAS 번호: (성분명과 확인된 CAS 번호 기재)
2. 규제 상태 판정: (🟢 안전 / 🟡 주의 / 🔴 위험-PFAS규제대상 중 택1)
3. 화공/반도체 공정 관점의 한 줄 설명: (판정 근거 및 대체 물질 필요 여부 등)
"""

        response = model.generate_content([prompt, image])

        # 결과 출력
        st.success("분석 완료!")
        st.markdown(response.text)

    except Exception as e:
        st.error(f"오류가 발생했습니다: {e}")
