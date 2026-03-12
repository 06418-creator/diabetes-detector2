import streamlit as st
from ultralytics import YOLO
from PIL import Image

# --- 1. หน้าจอแบบ Wide เพื่อให้มีพื้นที่ด้านข้างมากขึ้น ---
st.set_page_config(page_title="Smart Urine Analyzer", page_icon="🧬", layout="wide")

# --- 2. CSS ปรับแต่งให้ทุกอย่างชิดกัน (Compact Design) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;500;700&display=swap');
    html, body, [class*="css"] { font-family: 'Prompt', sans-serif; }
    #MainMenu, footer, header {visibility: hidden;}
    
    /* ลดระยะห่างด้านบนของหน้าเว็บ */
    .block-container { padding-top: 2rem; padding-bottom: 0rem; }

    /* ปุ่มกดแบบ Compact */
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        background: linear-gradient(135deg, #2563EB, #3B82F6);
        color: white;
        padding: 10px;
        font-weight: 700;
        border: none;
        box-shadow: 0 4px 10px rgba(59, 130, 246, 0.3);
    }

    /* ผลลัพธ์แบบ Card เล็กกะทัดรัด */
    .result-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
        text-align: center;
        border: 1px solid #e2e8f0;
        margin-top: 10px;
    }
    
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 15px;
        font-size: 16px;
        font-weight: 700;
    }
    .neg { background: #ECFDF5; color: #059669; }
    .trace { background: #FFFBEB; color: #D97706; }
    .plus { background: #FEF2F2; color: #DC2626; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. ส่วนหัวแบบบรรทัดเดียว ---
col_h1, col_h2 = st.columns([1, 8])
with col_h1: st.markdown("## 🧬")
with col_h2: st.markdown("### Smart Urine Analyzer <small style='font-size:14px; color:gray;'>| Diabetes Screening</small>", unsafe_allow_html=True)

# --- 4. Load Model ---
@st.cache_resource
def load_model():
    try: return YOLO('best (5).pt') 
    except: return None

model = load_model()
class_names = ['Neg', 'Trace', 'plus1', 'plus2', 'plus3', 'plus4']

# --- 5. Main Layout (แบ่ง 3 คอลัมน์เพื่อให้จบในหน้าเดียว) ---
col_input, col_preview, col_result = st.columns([1.5, 1.2, 1.8])

with col_input:
    st.markdown("**1. Upload Image**")
    uploaded_file = st.file_uploader("", type=["jpg", "png"], label_visibility="collapsed")
    if uploaded_file:
        st.success("File uploaded!")
        analyze_btn = st.button('✨ ANALYZE NOW')
    else:
        analyze_btn = False

if uploaded_file:
    image = Image.open(uploaded_file)
    
    with col_preview:
        st.markdown("**2. Preview**")
        # จำกัดความสูงของรูป (Height) เพื่อไม่ให้ล้นจอ แต่ยังเห็นทั้งแถบ
        st.image(image, use_container_width=True)

    with col_result:
        st.markdown("**3. Result**")
        if analyze_btn and model:
            with st.spinner('Thinking...'):
                results = model(image)
                if len(results[0].boxes) > 0:
                    box = results[0].boxes[0]
                    label = class_names[int(box.cls[0])]
                    conf = float(box.conf[0])
                    
                    # กำหนดสี Badge
                    b_class = "neg" if label == 'Neg' else "trace" if label == 'Trace' else "plus"
                    
                    # แสดงรูปที่ตรวจจับแล้ว (ขนาดเล็ก)
                    st.image(results[0].plot(), use_container_width=True)
                    
                    # แสดงการวิเคราะห์แบบกระชับ
                    st.markdown(f"""
                        <div class="result-card">
                            <div class="badge {b_class}">{label.upper()}</div>
                            <p style="margin: 5px 0; font-size: 14px; color: #64748B;">Confidence: {conf:.1%}</p>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.error("No strip detected.")
        else:
            st.info("Waiting for analysis...")

st.markdown("---")
st.caption("Disclaimer: For preliminary screening only.")
