import streamlit as st
from ultralytics import YOLO
from PIL import Image

# --- 1. ตั้งค่าหน้าจอ ---
st.set_page_config(page_title="Smart Urine Analyzer", page_icon="🧬", layout="centered")

# --- 2. Premium CSS Style ---
st.markdown("""
    <style>
    /* นำเข้าฟอนต์ Google Fonts เพื่อความอินเตอร์ */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=Prompt:wght@300;500;700&display=swap');

    /* ปรับฟอนต์ทั้งหน้าเว็บ */
    html, body, [class*="css"]  {
        font-family: 'Prompt', 'Inter', sans-serif;
    }

    /* ซ่อน Header/Footer ของ Streamlit ให้ดูเป็นแอปส่วนตัว */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* ตกแต่งปุ่มกด (Gradient Button) */
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        background: linear-gradient(135deg, #2563EB, #3B82F6);
        color: white;
        padding: 12px 20px;
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
        border: none;
    }

    /* กล่องแสดงผลลัพธ์สไตล์ Premium Card */
    .premium-card {
        background: #ffffff;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
        text-align: center;
        margin-top: 20px;
        margin-bottom: 20px;
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

    /* สีตามระดับอาการ (ดูหรูหราขึ้น ไม่จ้าเกินไป) */
    .neg-badge { background: #ECFDF5; color: #059669; border: 1px solid #10B981;}
    .trace-badge { background: #FFFBEB; color: #D97706; border: 1px solid #F59E0B;}
    .plus-badge { background: #FEF2F2; color: #DC2626; border: 1px solid #EF4444;}
    
    /* กล่องคำแนะนำอัปโหลด */
    .upload-instruction {
        background-color: #F8FAFC; 
        padding: 15px; 
        border-radius: 10px; 
        border-left: 4px solid #3B82F6; 
        margin-bottom: 15px;
        color: #1E293B; 
        font-size: 14px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. ส่วนหัว (Header) สไตล์แอป พร้อมโลโก้ ---
col1, col2 = st.columns([1, 4])
with col1:
    # โหลดโลโก้ (ถ้าหาไฟล์ไม่เจอ จะใช้ Emoji รูป DNA แทน)
    try:
        logo_img = Image.open("logo_diagnostic.png") 
        st.image(logo_img, use_container_width=True)
    except:
        st.markdown("<h1 style='text-align: center; color: #1E293B; font-size: 50px; margin-top: -10px;'>🧬</h1>", unsafe_allow_html=True)

with col2:
    st.markdown("<h2 style='text-align: left; color: #1E293B; font-weight: 800; margin-top: -10px;'>Smart Urine Analyzer</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: left; color: #64748B; font-size: 16px; margin-top: -10px;'>AI-Powered Diabetes Screening Tool</p>", unsafe_allow_html=True)

st.markdown("---")

# --- 4. โหลดโมเดล ---
@st.cache_resource
def load_model():
    try:
        return YOLO('best (4).pt') 
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
st.markdown("<div class='upload-instruction'><b>Instruction:</b> กรุณาอัปโหลดรูปภาพแผ่นตรวจปัสสาวะ โดยให้เห็นแถบสีที่ 10 (Glucose) อย่างชัดเจนเพื่อผลลัพธ์ที่แม่นยำ</div>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose an image file...", type=["jpg", "jpeg", "png"])

# --- 7. ประมวลผลและวิเคราะห์ ---
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Image ready for analysis', use_container_width=True)
    
    if st.button('✨ ANALYZE NOW'):
        if model is not None:
            with st.spinner('AI is analyzing the pad...'):
                results = model(image)
                
                if len(results[0].boxes) > 0:
                    top_box = results[0].boxes[0]
                    cls_id = int(top_box.cls[0])
                    conf = float(top_box.conf[0])
                    
                    label = class_names[cls_id]
                    info = class_info[label]
                    
                    st.image(results[0].plot(), use_container_width=True)
                    
                    # UI ผลลัพธ์แบบ Premium Card
                    res_html = f"""
                        <div class="premium-card">
                            <h4 style="color: #64748B; margin-bottom: 15px; font-size: 16px;">ANALYSIS RESULT</h4>
                            <div class="status-badge {info['badge']}">
                                {info['name']}
                            </div>
                            <p style="color: #94A3B8; font-size: 14px; margin-top: 10px;">
                                AI Confidence Level: <b>{conf:.1%}</b>
                            </p>
                            <hr style="border: none; border-top: 1px solid #F1F5F9; margin: 20px 0;">
                            <p style="color: #334155; font-size: 16px; line-height: 1.6;">
                                <b>Medical Advice:</b><br>{info['desc']}
                            </p>
                        </div>
                    """
                    st.markdown(res_html, unsafe_allow_html=True)
                else:
                    st.warning("⚠️ No glucose pad detected. Please check the image and try again.")
        else:
            st.error("System Error: Model not found.")

st.markdown("<div style='text-align: center; margin-top: 40px; color: #CBD5E1; font-size: 12px;'>Disclaimer: This tool provides AI-based estimations. Always consult a healthcare professional for accurate medical diagnosis.</div>", unsafe_allow_html=True)
