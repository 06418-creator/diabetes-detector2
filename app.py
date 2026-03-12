import streamlit as st
from ultralytics import YOLO
from PIL import Image

# --- 1. ตั้งค่าหน้าจอ ---
st.set_page_config(page_title="Smart Urine Analyzer", page_icon="🧬", layout="centered")

# --- 2. Premium CSS Style ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=Prompt:wght@300;500;700&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Prompt', 'Inter', sans-serif;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* ตกแต่งปุ่มกด */
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
        margin-top: 10px;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.6);
        color: white;
    }

    /* Premium Card */
    .premium-card {
        background: #ffffff;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
        text-align: center;
        margin-top: 20px;
        border: 1px solid #f0f0f0;
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
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. ส่วนหัว (Header) ---
col_logo, col_title = st.columns([1, 4])
with col_logo:
    try:
        logo_img = Image.open("logo_diagnostic.png") 
        st.image(logo_img, width=80)
    except:
        st.markdown("<h1 style='margin-top: 0;'>🧬</h1>", unsafe_allow_html=True)

with col_title:
    st.markdown("<h2 style='margin-bottom: 0;'>Smart Urine Analyzer</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #64748B;'>Development for Preliminary Diabetes Screening</p>", unsafe_allow_html=True)

st.markdown("---")

# --- 4. โหลดโมเดล ---
@st.cache_resource
def load_model():
    try:
        return YOLO('best (5).pt') 
    except Exception as e:
        st.error(f"System Error: Unable to load AI model. ({e})")
        return None

model = load_model()

# --- 5. ข้อมูลคลาส ---
class_info = {
    'Neg': {'name': 'NEGATIVE (ปกติ)', 'badge': 'neg-badge', 'desc': 'ไม่พบความผิดปกติของระดับน้ำตาลในปัสสาวะ'},
    'Trace': {'name': 'TRACE (พบเล็กน้อย)', 'badge': 'trace-badge', 'desc': 'พบน้ำตาลปริมาณน้อยมาก ควรเฝ้าระวังและตรวจซ้ำ'},
    'plus1': {'name': '+1 (ระดับเริ่มต้น)', 'badge': 'plus-badge', 'desc': 'พบน้ำตาล ~100 mg/dL ควรปรับพฤติกรรมการบริโภค'},
    'plus2': {'name': '+2 (ระดับปานกลาง)', 'badge': 'plus-badge', 'desc': 'พบน้ำตาล ~250 mg/dL แนะนำให้ปรึกษาแพทย์'},
    'plus3': {'name': '+3 (ระดับสูง)', 'badge': 'plus-badge', 'desc': 'พบน้ำตาล ~500 mg/dL มีความเสี่ยงสูง ควรพบแพทย์'},
    'plus4': {'name': '+4 (ระดับสูงมาก)', 'badge': 'plus-badge', 'desc': 'พบน้ำตาล >1000 mg/dL อันตราย! โปรดพบแพทย์ทันที'}
}
class_names = ['Neg', 'Trace', 'plus1', 'plus2', 'plus3', 'plus4']

# --- 6. ส่วนอัปโหลดรูปภาพ ---
st.markdown("<div class='upload-instruction'><b>คำแนะนำ:</b> อัปโหลดรูปแผ่นตรวจที่เห็นแถบสีชัดเจน เพื่อประสิทธิภาพสูงสุดของ AI</div>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("เลือกไฟล์รูปภาพ...", type=["jpg", "jpeg", "png"])

# --- 7. ส่วนประมวลผล (Layout ใหม่ ไม่ต้องเลื่อนจอ) ---
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    # แบ่งคอลัมน์เพื่อให้ Preview เล็ก และปุ่มอยู่ใกล้กัน
    col_img, col_action = st.columns([1, 1])
    
    with col_img:
        st.image(image, caption='รูปที่อัปโหลด', width=240) # จำกัดขนาดรูปให้เล็กลง
    
    with col_action:
        st.write("### ") # เพิ่มระยะเว้นว่าง
        analyze_btn = st.button('✨ ANALYZE NOW')
        st.info("กดปุ่มเพื่อเริ่มวิเคราะห์")

    if analyze_btn:
        if model is not None:
            with st.spinner('กำลังวิเคราะห์...'):
                results = model(image)
                
                if len(results[0].boxes) > 0:
                    top_box = results[0].boxes[0]
                    cls_id = int(top_box.cls[0])
                    conf = float(top_box.conf[0])
                    
                    label = class_names[cls_id]
                    info = class_info[label]
                    
                    # แสดงผลลัพธ์
                    st.markdown("---")
                    st.image(results[0].plot(), caption='ผลการตรวจจับโดย AI', use_container_width=True)
                    
                    st.markdown(f"""
                        <div class="premium-card">
                            <h4 style="color: #64748B; margin-bottom: 10px; font-size: 14px;">ผลการวิเคราะห์</h4>
                            <div class="status-badge {info['badge']}">
                                {info['name']}
                            </div>
                            <p style="color: #94A3B8; font-size: 13px;">AI Confidence: <b>{conf:.1%}</b></p>
                            <hr style="border: none; border-top: 1px solid #F1F5F9; margin: 15px 0;">
                            <p style="color: #1E293B; font-size: 16px;">
                                <b>คำแนะนำ:</b> {info['desc']}
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning("⚠️ ไม่พบแถบตรวจปัสสาวะในรูปภาพ กรุณาลองใหม่")

st.markdown("<br><div style='text-align: center; color: #CBD5E1; font-size: 11px;'>หมายเหตุ: ระบบนี้เป็นเพียงการคัดกรองเบื้องต้น โปรดปรึกษาแพทย์เพื่อรับการวินิจฉัยที่ถูกต้อง</div>", unsafe_allow_html=True)
