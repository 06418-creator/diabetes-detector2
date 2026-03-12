import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os

# --- 1. ตั้งค่าหน้าจอ (Wide Mode) ---
st.set_page_config(page_title="Smart Urine Analyzer", page_icon="🧬", layout="wide")

# --- 2. CSS ระดับพรีเมียม (Premium Styling) ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Prompt', sans-serif; }
#MainMenu, footer, header {visibility: hidden;}
.block-container { padding-top: 1.5rem; padding-bottom: 0rem; max-width: 1200px; }

/* ตกแต่ง Header ให้ดูคลีน */
.header-box {
    border-bottom: 2px solid #f1f5f9;
    padding-bottom: 10px;
    margin-bottom: 20px;
}

/* ปุ่มวิเคราะห์แบบไล่สี หรูหรา */
.stButton>button {
    width: 100%;
    height: 55px;
    border-radius: 12px;
    background: linear-gradient(135deg, #2563EB, #3B82F6);
    color: white;
    font-size: 16px;
    font-weight: 600;
    border: none;
    box-shadow: 0 4px 15px rgba(37, 99, 235, 0.3);
    transition: all 0.3s ease;
    margin-top: 30px; /* ดันปุ่มให้ลงมาอยู่กึ่งกลางรูป */
}
.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(37, 99, 235, 0.4);
}

/* การ์ดผลลัพธ์สไตล์ Glassmorphism */
.premium-card {
    background: #ffffff;
    border-radius: 16px;
    padding: 24px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.05);
    border: 1px solid #e2e8f0;
}

/* ป้ายกำกับผลลัพธ์ไล่สี */
.status-badge {
    display: inline-block;
    padding: 8px 24px;
    border-radius: 30px;
    font-size: 18px;
    font-weight: 700;
    color: white;
    margin-bottom: 15px;
    letter-spacing: 0.5px;
}
.bg-neg { background: linear-gradient(135deg, #10B981, #059669); }
.bg-trace { background: linear-gradient(135deg, #F59E0B, #D97706); }
.bg-plus { background: linear-gradient(135deg, #EF4444, #DC2626); }

/* กล่องคำแนะนำ */
.advice-box {
    background: #f8fafc;
    border-left: 4px solid #3b82f6;
    padding: 15px 20px;
    border-radius: 0 12px 12px 0;
    margin-top: 15px;
}
</style>
""", unsafe_allow_html=True)

# --- 3. ส่วนหัว (Header) ---
st.markdown("""
<div class="header-box">
    <h2 style="margin:0; color:#1e293b;">🧬 Smart Urine Analyzer</h2>
    <span style="color:#64748b; font-size:15px;">ระบบคัดกรองความเสี่ยงโรคเบาหวานผ่านแถบตรวจปัสสาวะอัจฉริยะ</span>
</div>
""", unsafe_allow_html=True)

# --- 4. โหลดโมเดล & ฐานข้อมูล ---
model_file = 'best (5).pt'

@st.cache_resource
def load_model():
    try: return YOLO(model_file) 
    except: return None

class_names = ['Neg', 'Trace', 'plus1', 'plus2', 'plus3', 'plus4']
class_info = {
    'Neg': {'name': 'NEGATIVE (ปกติ)', 'color': 'bg-neg', 'desc': 'ไม่พบน้ำตาลในปัสสาวะ ระดับน้ำตาลอยู่ในเกณฑ์ปกติ แนะนำให้ดูแลสุขภาพและออกกำลังกายอย่างสม่ำเสมอ'},
    'Trace': {'name': 'TRACE (พบเล็กน้อย)', 'color': 'bg-trace', 'desc': 'พบน้ำตาลปริมาณน้อยมาก (เริ่มมีความเสี่ยง) ควรเฝ้าระวัง ลดการบริโภคของหวาน และตรวจซ้ำในภายหลัง'},
    'plus1': {'name': '+1 (ระดับเริ่มต้น)', 'color': 'bg-plus', 'desc': 'พบน้ำตาล ~100 mg/dL ควรปรับพฤติกรรมการทานอาหาร ลดแป้งและน้ำตาลโดยด่วน'},
    'plus2': {'name': '+2 (ระดับปานกลาง)', 'color': 'bg-plus', 'desc': 'พบน้ำตาล ~250 mg/dL มีความเสี่ยงเป็นโรคเบาหวาน ควรปรึกษาแพทย์เพื่อเจาะเลือดตรวจอย่างละเอียด'},
    'plus3': {'name': '+3 (ระดับสูง)', 'color': 'bg-plus', 'desc': 'พบน้ำตาล ~500 mg/dL มีความเสี่ยงสูงต่อภาวะแทรกซ้อน <b>โปรดไปพบแพทย์เพื่อรับการรักษาทันที</b>'},
    'plus4': {'name': '+4 (ระดับสูงมาก)', 'color': 'bg-plus', 'desc': 'พบน้ำตาล >1000 mg/dL 🚨 <b>อันตรายร้ายแรง!</b> อาจเกิดภาวะช็อก <b>โปรดไปโรงพยาบาลฉุกเฉินทันที</b>'}
}

# --- 5. Layout หลัก (ซ้าย: นำเข้าข้อมูล | ขวา: แสดงผล) ---
col_left, col_right = st.columns([1.5, 2.5], gap="large")

with col_left:
    st.markdown("<h4 style='color:#334155; font-size:16px;'>1. อัปโหลดรูปภาพแถบตรวจ</h4>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    
    analyze_now = False
    if uploaded_file:
        image = Image.open(uploaded_file)
        
        # แบ่งคอลัมน์ย่อย: รูปอยู่ซ้าย ปุ่มอยู่ขวา
        col_img, col_btn = st.columns([1, 1.2])
        with col_img:
            # ล็อกขนาดรูป ไม่ให้หน้าจอยืด
            st.image(image, width=160, caption="รูปที่อัปโหลด")
        with col_btn:
            analyze_now = st.button('✨ วิเคราะห์ผล (ANALYZE)')

with col_right:
    st.markdown("<h4 style='color:#334155; font-size:16px;'>2. ผลการประมวลผลด้วย AI</h4>", unsafe_allow_html=True)
    
    # ตรวจสอบไฟล์โมเดลก่อนทำงาน
    if not os.path.exists(model_file):
        st.error(f"❌ ระบบไม่พบไฟล์โมเดล '{model_file}'")
        st.info("💡 กรุณานำไฟล์ 'best (5).pt' มาวางในโฟลเดอร์เดียวกับโค้ดนี้ครับ")
        
    elif uploaded_file and analyze_now:
        model = load_model()
        if model:
            with st.spinner('🤖 AI กำลังประมวลผล...'):
                results = model(image)
                
                if len(results[0].boxes) > 0:
                    box = results[0].boxes[0]
                    label = class_names[int(box.cls[0])]
                    conf = float(box.conf[0])
                    info = class_info[label]
                    
                    # แบ่งคอลัมน์ย่อยสำหรับผลลัพธ์: รูป AI วงกล่องอยู่ซ้าย, การ์ดข้อความอยู่ขวา
                    col_res_img, col_res_text = st.columns([1, 2.2])
                    
                    with col_res_img:
                        st.image(results[0].plot(), width=160, caption="AI Detection")
                        
                    with col_res_text:
                        # การ์ดแสดงผลแบบพรีเมียม (HTML/CSS)
                        st.markdown(f"""
                            <div class="premium-card">
                                <div class="status-badge {info['color']}">{info['name']}</div>
                                <div style="color: #64748b; font-size: 15px; margin-bottom: 10px;">
                                    🎯 ความแม่นยำ (Confidence): <strong style="color:#0f172a;">{conf:.1%}</strong>
                                </div>
                                
                                <div class="advice-box">
                                    <h5 style="margin:0 0 5px 0; color:#1e293b; font-size:14px;">💡 คำแนะนำทางการแพทย์เบื้องต้น</h5>
                                    <p style="margin:0; color:#475569; font-size:14px; line-height:1.6;">
                                        {info['desc']}
                                    </p>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                else:
                    st.warning("⚠️ AI ค้นหาแถบสีไม่พบ กรุณาครอบรูปให้กระชับและเห็นแถบสีชัดเจนขึ้นครับ")
                    
    elif uploaded_file:
        st.info("👈 กดปุ่ม '✨ วิเคราะห์ผล' ที่อยู่ข้างๆ รูปภาพฝั่งซ้าย เพื่อเริ่มต้นการทำงาน")
    else:
        st.markdown("""
            <div style="background:#f1f5f9; padding:30px; border-radius:12px; text-align:center; color:#64748b; border: 2px dashed #cbd5e1;">
                📸 กรุณาอัปโหลดรูปภาพที่ฝั่งซ้าย เพื่อดูผลการวิเคราะห์
            </div>
        """, unsafe_allow_html=True)
