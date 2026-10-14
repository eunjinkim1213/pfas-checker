import io
import json
import re

import pandas as pd
import streamlit as st
from google import genai
from google.genai import errors, types
from PIL import Image

# ──────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────
# 앞에서부터 차례로 시도합니다. 429(한도 초과)/404(모델 없음)면 다음 모델로 넘어갑니다.
# 실제 사용 가능한 모델명으로 바꿔 쓰세요 (AI Studio에서 확인).
MODELS = [
    "gemini-3.8-flash",
    "gemini-3.5-flash-lite",
]
MAX_IMAGE_SIDE = 1024  # 이미지 축소 (토큰/전송량 절약)

# 불소계 키워드 ("무불소", "PFC-Free" 등은 먼저 제거한 뒤 검사)
PFAS_KEYWORDS = [
    "불소", "불소수지", "불소계", "ptfe", "fluoro", "pfas",
    "pfoa", "pfos", "perfluoro", "polyfluoro",
]
FLUORINE_FREE_PATTERNS = ["무불소", "불소 무함유", "pfc-free", "pfc free", "fluorine-free", "fluorine free"]

# 고위험 성분 / 주의 성분 키워드
STRONG_HAZARD_KEYWORDS = [
    "차아염소산나트륨", "수산화나트륨", "sodium hypochlorite", "sodium hydroxide",
    "가성소다", "염산", "hydrochloric acid", "황산", "sulfuric acid",
]
CAUTION_KEYWORDS = [
    "에탄올", "ethanol", "이소프로필", "isopropyl", "아세톤", "acetone",
    "톨루엔", "toluene", "자일렌", "xylene", "메탄올", "methanol", "유기용제", "solvent",
]

# 픽토그램 분류
RED_PICTOGRAMS = {"corrosive", "skull", "environment"}
YELLOW_PICTOGRAMS = {"flame", "exclamation", "health_hazard", "oxidizer", "explosive", "gas_cylinder"}
PICTO_LABEL = {
    "corrosive": "부식성", "skull": "급성독성", "environment": "환경유해성",
    "flame": "인화성", "exclamation": "자극성/경고", "health_hazard": "건강유해성",
    "oxidizer": "산화성", "explosive": "폭발성", "gas_cylinder": "고압가스",
}

EXTRACT_PROMPT = """너는 화학제품 라벨 판독 도우미야. 사진 속 라벨을 읽고 아래 JSON만 출력해. 설명, 마크다운 금지.

{
  "readable": true/false,
  "product_name": "제품명 (없으면 빈 문자열)",
  "ingredients": [{"name": "성분명 (보이는 그대로)", "cas": "CAS 번호 (없으면 빈 문자열)"}],
  "pictograms": ["corrosive","skull","environment","flame","exclamation","health_hazard","oxidizer","explosive","gas_cylinder" 중 라벨에 보이는 것만],
  "signal_word": "위험/경고/없음"
}

규칙: 라벨에 없는 내용은 추측하지 마. 읽을 수 없으면 readable을 false로."""


# ──────────────────────────────────────────────
# 에러 정의
# ──────────────────────────────────────────────
class QuotaExhausted(Exception):
    """모든 모델의 무료 한도가 소진됨"""


# ──────────────────────────────────────────────
# 유틸
# ──────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_db(path: str = "pfas_list.csv") -> pd.DataFrame:
    return pd.read_csv(path, dtype=str).fillna("")


def normalize_cas(value: str) -> str:
    return re.sub(r"[^0-9\-]", "", value or "")


def prepare_image(raw: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(raw))
    img = img.convert("RGB")
    img.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
    return img


def parse_json(text: str) -> dict:
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    return json.loads(cleaned)


# ──────────────────────────────────────────────
# Gemini 호출 (요청 1회 + 모델 폴백 + 이미지 해시 캐시)
# ──────────────────────────────────────────────
def call_with_fallback(client: genai.Client, image: Image.Image) -> dict:
    config = types.GenerateContentConfig(response_mime_type="application/json", temperature=0)
    last_error = None
    for name in MODELS:
        try:
            resp = client.models.generate_content(
                model=name, contents=[EXTRACT_PROMPT, image], config=config
            )
            return parse_json(resp.text)
        except errors.APIError as e:
            # 429: 한도 초과, 404: 모델 없음 → 다음 모델 시도
            if getattr(e, "code", None) in (429, 404):
                last_error = e
                continue
            raise
        except json.JSONDecodeError as e:
            last_error = e
            continue
    if isinstance(last_error, errors.APIError) and getattr(last_error, "code", None) == 429:
        raise QuotaExhausted() from last_error
    raise RuntimeError(f"모든 모델 호출에 실패했습니다: {last_error}")


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def extract_label(image_bytes: bytes, api_key: str) -> dict:
    """같은 이미지는 서버 전체에서 한 번만 API 호출 (예외는 캐시되지 않음)."""
    client = genai.Client(api_key=api_key)
    return call_with_fallback(client, prepare_image(image_bytes))


# ──────────────────────────────────────────────
# 판정 로직 (AI가 아니라 코드가 판정)
# ──────────────────────────────────────────────
def match_db(ingredients: list[dict], db: pd.DataFrame) -> list[dict]:
    cas_cols = [c for c in db.columns if "cas" in c.lower()]
    name_cols = [c for c in db.columns if c not in cas_cols]

    cas_index = {}
    for _, row in db.iterrows():
        for c in cas_cols:
            key = normalize_cas(row[c])
            if key:
                cas_index[key] = row.to_dict()

    name_index = []
    for _, row in db.iterrows():
        for c in name_cols:
            val = str(row[c]).strip().lower()
            if len(val) >= 4:
                name_index.append((val, row.to_dict()))

    hits = []
    for ing in ingredients:
        name = (ing.get("name") or "").strip()
        cas = normalize_cas(ing.get("cas", ""))
        name_l = name.lower()

        if cas and cas in cas_index:
            hits.append({"ingredient": name, "cas": cas, "by": "CAS 일치", "row": cas_index[cas]})
            continue
        for db_name, row in name_index:
            if db_name in name_l or (len(name_l) >= 4 and name_l in db_name):
                hits.append({"ingredient": name, "cas": cas, "by": "성분명 일치", "row": row})
                break
    return hits


def contains_any(text: str, keywords: list[str]) -> list[str]:
    t = text.lower()
    return [k for k in keywords if k.lower() in t]


def judge(label: dict, db: pd.DataFrame) -> dict:
    ingredients = label.get("ingredients") or []
    pictos = {p for p in (label.get("pictograms") or []) if isinstance(p, str)}

    reasons_red, reasons_yellow = [], []

    # 1) 규제 DB 매칭
    hits = match_db(ingredients, db)
    for h in hits:
        reasons_red.append(f"규제 DB 해당 물질: **{h['ingredient']}** ({h['by']})")

    # 2) 불소계 키워드 (무불소 표기는 제외하고 검사)
    for ing in ingredients:
        name = (ing.get("name") or "")
        cleaned = name.lower()
        for pat in FLUORINE_FREE_PATTERNS:
            cleaned = cleaned.replace(pat, "")
        found = contains_any(cleaned, PFAS_KEYWORDS)
        if found:
            reasons_red.append(f"불소계 의심 성분: **{name}**")

        if contains_any(name, STRONG_HAZARD_KEYWORDS):
            reasons_red.append(f"강한 유해 화학물질: **{name}**")
        elif contains_any(name, CAUTION_KEYWORDS):
            reasons_yellow.append(f"인화성/유기용제 계열 성분: **{name}**")

    # 3) 픽토그램
    for p in sorted(pictos & RED_PICTOGRAMS):
        reasons_red.append(f"위험 픽토그램: {PICTO_LABEL.get(p, p)}")
    for p in sorted(pictos & YELLOW_PICTOGRAMS):
        reasons_yellow.append(f"주의 픽토그램: {PICTO_LABEL.get(p, p)}")

    if reasons_red:
        level = "red"
    elif reasons_yellow:
        level = "yellow"
    else:
        level = "green"

    return {"level": level, "red": reasons_red, "yellow": reasons_yellow, "hits": hits}


# ──────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────
st.set_page_config(page_title="EU PFAS Checker", page_icon="🧪")
st.title("🧪 EU REACH·PFAS 화학성분 판별기")
st.write("화학제품 라벨이나 성분표를 사진으로 찍어 올리면, AI가 라벨을 읽고 규제 DB와 대조합니다.")

# API 키
api_key = st.secrets.get("GEMINI_API_KEY") if "GEMINI_API_KEY" in st.secrets else None
if not api_key:
    api_key = st.sidebar.text_input("Gemini API Key를 입력하세요", type="password")

# 규제 DB
try:
    db = load_db()
except Exception:
    st.error("pfas_list.csv 파일을 불러올 수 없습니다.")
    st.stop()

if not api_key:
    st.info("API 키가 필요합니다. (Secrets 또는 사이드바에 입력)")
    st.stop()

# 입력 방식 선택 (카메라는 선택했을 때만 켜짐)
mode = st.radio("입력 방식", ["📷 카메라 촬영", "🖼️ 파일 업로드"], horizontal=True)
if mode.startswith("📷"):
    uploaded = st.camera_input("라벨 사진을 촬영하세요")
else:
    uploaded = st.file_uploader("사진 파일을 올리세요", type=["jpg", "jpeg", "png"])

if uploaded:
    image_bytes = uploaded.getvalue()
    st.image(image_bytes, caption="분석할 라벨 사진", use_container_width=True)

    # 버튼 한 번 = API 요청 최대 한 번 (같은 이미지는 캐시)
    if st.button("🔍 규제 성분 분석 시작", type="primary"):
        with st.spinner("AI가 라벨을 읽는 중입니다..."):
            try:
                st.session_state["label"] = extract_label(image_bytes, api_key)
                st.session_state["label_key"] = hash(image_bytes)
            except QuotaExhausted:
                st.session_state.pop("label", None)
                st.warning("오늘 무료 분석 한도가 모두 소진됐어요. 내일 다시 시도해 주세요.")
            except Exception as e:
                st.session_state.pop("label", None)
                st.error(f"오류가 발생했습니다: {e}")

    # 다른 사진으로 바뀌면 이전 결과 숨김
    if st.session_state.get("label_key") != hash(image_bytes):
        st.session_state.pop("label", None)

label = st.session_state.get("label")
if uploaded and label:
    if not label.get("readable", True):
        st.warning("라벨을 읽기 어렵습니다. 더 밝고 선명하게 다시 촬영해 주세요.")
    else:
        result = judge(label, db)

        if result["level"] == "red":
            st.error("🔴 위험 (PFAS 규제대상 / 고위험)")
        elif result["level"] == "yellow":
            st.warning("🟡 주의")
        else:
            st.success("🟢 안전 (감지된 위험 요소 없음)")

        reasons = result["red"] + result["yellow"]
        if reasons:
            st.markdown("**판정 근거**")
            for r in dict.fromkeys(reasons):  # 중복 제거, 순서 유지
                st.markdown(f"- {r}")

        if label.get("product_name"):
            st.caption(f"제품명: {label['product_name']}")

        ingredients = label.get("ingredients") or []
        if ingredients:
            st.markdown("**인식된 성분**")
            st.dataframe(
                pd.DataFrame(ingredients).rename(columns={"name": "성분명", "cas": "CAS 번호"}),
                use_container_width=True,
                hide_index=True,
            )

        if result["hits"]:
            with st.expander("규제 DB 매칭 상세"):
                st.dataframe(pd.DataFrame([h["row"] for h in result["hits"]]), use_container_width=True)

        st.caption("※ AI의 라벨 판독에는 오류가 있을 수 있습니다. 중요한 판단은 SDS(물질안전보건자료)로 재확인하세요.")
