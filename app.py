import streamlit as st
from ultralytics import YOLO
from PIL import Image

# --- 1. ตั้งค่าหน้าจอ ---
st.set_page_config(page_title="Smart Urine Analyzer", page_icon="🧬", layout="centered")

# --- 2. Premium CSS Style ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=Prompt:wght@300;500;700&display=swap');
    html, body, [class*="css"]  { font-family: 'Prompt', 'Inter', sans-serif; }
    #MainMenu, footer, header {visibility: hidden;}

    .stButton>button {
        width: 100%;
        border-radius: 12px;
        background: linear-gradient(135deg, #2563EB, #3B82F6);
        color: white;
        padding: 15px 20px;
        font-size: 18px;
        font-weight: 700;
        border: none;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.4);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.6);
        color: white;
    }

    .premium-card {
        background: #ffffff;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
        text-align: center;
        margin-top: 20px;
        border: 1px solid #f0f0f0;
        color: #1E293B;
    }
    
    .status-badge {
        display: inline-block;
        padding: 8px 16px;
        border-radius: 20px;
        font-size: 20px;
        font-weight: 700;
        margin-bottom: 12px;
    }
    .neg-badge { background: #ECFDF5; color: #059669; border: 1px solid #10B981;}
    .trace-badge { background: #FFFBEB; color: #D97706; border: 1px solid #F59E0B;}
    .plus-badge { background: #FEF2F2; color: #DC2626; border: 1px solid #EF4444;}
    
    .upload-instruction {
        background-color: #F8FAFC; 
        padding: 15px; 
        border-radius: 10px; 
        border-left: 4px solid #3B82F6; 
        margin-bottom: 15px;
        font-size: 14px;
        color: #1E293B;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. Header ---
col_logo, col_title = st.columns([1, 5])
with col_logo:
    st.markdown("<h1 style='margin:0;'>🧬</h1>", unsafe_allow_html=True)
with col_title:
    st.markdown("<h2 style='margin:0;'>Smart Urine Analyzer</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #64748B; margin:0;'>Preliminary Diabetes Screening Tool</p>", unsafe_allow_html=True)

st.markdown("---")

# --- 4. Load Model ---
@st.cache_resource
def load_model():
    try:
        return YOLO('best (5).pt') 
    except Exception as e:
        st.error(f"Error: {e}")
        return None

model = load_model()
class_names = ['Neg', 'Trace', 'plus1', 'plus2', 'plus3', 'plus4']
class_info = {
    'Neg': {'name': 'NEGATIVE (ปกติ)', 'badge': 'neg-badge', 'desc': 'ไม่พบน้ำตาลในปัสสาวะ'},
    'Trace': {'name': 'TRACE (พบเล็กน้อย)', 'badge': 'trace-badge', 'desc': 'พบน้ำตาลปริมาณน้อยมาก'},
    'plus1': {'name': '+1 (ระดับเริ่มต้น)', 'badge': 'plus-badge', 'desc': 'พบน้ำตาล ~100 mg/dL'},
    'plus2': {'name': '+2 (ระดับปานกลาง)', 'badge': 'plus-badge', 'desc': 'พบน้ำตาล ~250 mg/dL'},
    'plus3': {'name': '+3 (ระดับสูง)', 'badge': 'plus-badge', 'desc': 'พบน้ำตาล ~500 mg/dL'},
    'plus4': {'name': '+4 (ระดับสูงมาก)', 'badge': 'plus-badge', 'desc': 'พบน้ำตาล >1000 mg/dL'}
}

# --- 5. Upload & Analysis ---
st.markdown("<div class='upload-instruction'><b>Instruction:</b> อัปโหลดรูปแผ่นตรวจที่เห็นแถบสีชัดเจน</div>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("เลือกไฟล์รูปภาพ...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    # แบ่ง Column: ซ้ายโชว์รูป (ขนาดพอดี), ขวามีปุ่ม Analyze
    col_left, col_right = st.columns([1.2, 1])
    
    with col_left:
        # ใช้ use_container_width=True เพื่อให้รูปขยายเต็มคอลัมน์ซ้าย แต่ไม่ใหญ่จนเกินหน้าจอ
        st.image(image, caption='Uploaded Strip', use_container_width=True)
    
    with col_right:
        st.write("## ") # เว้นระยะ
        analyze_btn = st.button('✨ ANALYZE NOW')
        if not analyze_btn:
            st.info("ตรวจสอบรูปภาพทางซ้าย แล้วกดปุ่มเพื่อเริ่มวิเคราะห์")

    if analyze_btn:
        if model is not None:
            with st.spinner('AI is analyzing...'):
                results = model(image)
                if len(results[0].boxes) > 0:
                    top_box = results[0].boxes[0]
                    cls_id = int(top_box.cls[0])
                    conf = float(top_box.conf[0])
                    info = class_info[class_names[cls_id]]
                    
                    st.markdown("---")
                    st.image(results[0].plot(), caption='AI Detection Result', use_container_width=True)
                    
                    st.markdown(f"""
                        <div class="premium-card">
                            <div class="status-badge {info['badge']}">{info['name']}</div>
                            <p style="color: #64748B;">AI Confidence: <b>{conf:.1%}</b></p>
                            <p><b>คำแนะนำ:</b> {info['desc']}</p>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning("⚠️ ไม่พบแถบตรวจปัสสาวะในรูปภาพ")

st.markdown("<br><div style='text-align: center; color: #CBD5E1; font-size: 11px;'>This tool is for preliminary screening only.</div>", unsafe_allow_html=True)
