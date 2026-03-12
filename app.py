import streamlit as st
from ultralytics import YOLO
from PIL import Image

# --- 1. หน้าจอแบบ Wide ---
st.set_page_config(page_title="Smart Urine Analyzer", page_icon="🧬", layout="wide")

# --- 2. CSS Compact Design ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;500;700&display=swap');
    html, body, [class*="css"] { font-family: 'Prompt', sans-serif; }
    #MainMenu, footer, header {visibility: hidden;}
    .block-container { padding-top: 1rem; }

    .stButton>button {
        width: 100%;
        border-radius: 10px;
        background: linear-gradient(135deg, #2563EB, #3B82F6);
        color: white;
        padding: 10px;
        font-weight: 700;
        border: none;
        margin-top: 10px;
    }
    .result-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
        text-align: center;
        border: 1px solid #e2e8f0;
    }
    .badge {
        display: inline-block;
        padding: 5px 15px;
        border-radius: 15px;
        font-size: 18px;
        font-weight: 700;
        margin-bottom: 5px;
    }
    .neg { background: #ECFDF5; color: #059669; }
    .trace { background: #FFFBEB; color: #D97706; }
    .plus { background: #FEF2F2; color: #DC2626; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. Header ---
st.markdown("### 🧬 Smart Urine Analyzer <small style='font-size:14px; color:gray;'>| ระบบคัดกรองเบาหวานอัจฉริยะ</small>", unsafe_allow_html=True)

# --- 4. Load Model ---
@st.cache_resource
def load_model():
    try:
        return YOLO('best (5).pt') 
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()
class_names = ['Neg', 'Trace', 'plus1', 'plus2', 'plus3', 'plus4']

# --- 5. Main Layout ---
# แบ่งเป็น 2 ส่วนใหญ่: ฝั่งซ้าย (จัดการรูป) | ฝั่งขวา (แสดงผลลัพธ์)
col_left, col_right = st.columns([1.5, 2])

with col_left:
    st.markdown("**📸 1. จัดการรูปภาพ**")
    uploaded_file = st.file_uploader("เลือกรูปแถบตรวจปัสสาวะ", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        # แสดงรูปตัวอย่างให้เห็นทั้งแถบ
        st.image(image, caption="รูปที่อัปโหลด", use_container_width=True)
        # วางปุ่ม Analyze ไว้ใต้รูปทันที
        analyze_now = st.button('✨ ANALYZE NOW')
    else:
        analyze_now = False

with col_right:
    st.markdown("**🎯 2. ผลการวิเคราะห์**")
    if uploaded_file and analyze_now:
        if model:
            with st.spinner('AI กำลังประมวลผล...'):
                results = model(image)
                if len(results[0].boxes) > 0:
                    # ดึงค่าผลลัพธ์
                    box = results[0].boxes[0]
                    cls_id = int(box.cls[0])
                    label = class_names[cls_id]
                    conf = float(box.conf[0])
                    
                    # เลือกสี Badge
                    b_style = "neg" if label == 'Neg' else "trace" if label == 'Trace' else "plus"
                    
                    # แสดงรูปที่ AI ตรวจจับได้
                    st.image(results[0].plot(), caption="AI Detection Result", use_container_width=True)
                    
                    # สรุปผล
                    st.markdown(f"""
                        <div class="result-card">
                            <h4 style="margin:0; color:#64748B; font-size:14px;">สรุปผลการตรวจ</h4>
                            <div class="badge {b_style}">{label.upper()}</div>
                            <p style="margin:0; font-size:14px;">ความแม่นยำ: {conf:.1%}</p>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.error("⚠️ AI หาแถบตรวจไม่เจอ กรุณาถ่ายรูปให้ชัดเจนขึ้น")
        else:
            st.error("❌ ไม่พบไฟล์โมเดล (best (5).pt)")
    elif uploaded_file:
        st.info("👆 ตรวจสอบรูปภาพแล้วกดปุ่ม 'ANALYZE NOW'")
    else:
        st.info("👈 กรุณาอัปโหลดรูปภาพที่ฝั่งซ้าย")

st.markdown("---")
st.caption("จัดทำโดย: [ชื่อของคุณ/กลุ่มของคุณ] | ความแม่นยำของโมเดล: 97.19%")
