import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os

# --- 1. ตั้งค่าหน้าจอ ---
st.set_page_config(page_title="Smart Urine Analyzer", page_icon="🧬", layout="wide")

# --- 2. CSS ระดับพรีเมียม ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Prompt', sans-serif; }
#MainMenu, footer, header {visibility: hidden;}
.block-container { padding-top: 1.5rem; padding-bottom: 0rem; max-width: 1100px; }

.stButton>button {
    width: 100%;
    height: 50px;
    border-radius: 10px;
    background: linear-gradient(135deg, #2563EB, #3B82F6);
    color: white;
    font-size: 16px;
    font-weight: 600;
    border: none;
    box-shadow: 0 4px 15px rgba(37, 99, 235, 0.25);
    transition: all 0.3s ease;
    margin-top: 15px; 
}
.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(37, 99, 235, 0.4);
}

/* CSS สำหรับการ์ดผลลัพธ์ */
.result-card {
    background: #ffffff;
    border-radius: 16px;
    padding: 24px;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.08);
    border: 1px solid #e2e8f0;
}
.badge {
    display: inline-block;
    padding: 8px 24px;
    border-radius: 30px;
    font-size: 18px;
    font-weight: 700;
    color: white;
    margin-bottom: 15px;
}
.bg-neg { background: linear-gradient(135deg, #10B981, #059669); }
.bg-trace { background: linear-gradient(135deg, #F59E0B, #D97706); }
.bg-plus { background: linear-gradient(135deg, #EF4444, #DC2626); }

.advice-box {
    background: #f8fafc;
    border-left: 5px solid #3b82f6;
    padding: 15px 20px;
    border-radius: 8px;
    margin-top: 15px;
}
</style>
""", unsafe_allow_html=True)

# --- 3. ส่วนหัว ---
st.markdown("## 🧬 Smart Urine Analyzer")
st.markdown("<span style='color:#64748b; font-size:16px;'>ระบบคัดกรองความเสี่ยงโรคเบาหวานผ่านแถบตรวจปัสสาวะอัจฉริยะ</span>", unsafe_allow_html=True)
st.markdown("---")

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

# --- 5. Layout (ซ้าย: อัปโหลด | ขวา: ผลลัพธ์) ---
col_left, col_right = st.columns([1.5, 2.5], gap="large")

with col_left:
    st.markdown("**1. อัปโหลดรูปภาพแถบตรวจ**")
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    
    analyze_now = False
    if uploaded_file:
        image = Image.open(uploaded_file)
        
        # จัดปุ่มให้อยู่ข้างรูปภาพ
        col_img, col_btn = st.columns([1, 1.2])
        with col_img:
            st.image(image, width=150)
        with col_btn:
            st.write("") # ดันปุ่มลงมานิดหน่อยให้สวยงาม
            analyze_now = st.button('✨ ANALYZE NOW')

with col_right:
    st.markdown("**2. ผลการวิเคราะห์**")
    
    if not os.path.exists(model_file):
        st.error(f"❌ ระบบไม่พบไฟล์โมเดล '{model_file}'")
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
                    
                    # แบ่งผลลัพธ์เป็น รูปภาพซ้าย / กล่องข้อความขวา
                    col_res_img, col_res_text = st.columns([1, 2.2])
                    
                    with col_res_img:
                        st.image(results[0].plot(), width=150)
                        
                    with col_res_text:
                        # [จุดสำคัญที่แก้บั๊ก] HTML ต้องชิดซ้ายสุด ห้ามมีช่องว่างนำหน้าเด็ดขาด!
                        html_str = f"""
<div class="result-card">
<div class="badge {info['color']}">{info['name']}</div>
<div style="font-size: 15px; color: #475569; margin-bottom: 10px;">
🎯 ความแม่นยำ (Confidence): <strong style="color: #0f172a;">{conf:.1%}</strong>
</div>
<div class="advice-box">
<h4 style="margin: 0 0 5px 0; color: #1e293b; font-size: 15px;">💡 คำแนะนำทางการแพทย์เบื้องต้น</h4>
<p style="margin: 0; color: #334155; font-size: 14px; line-height: 1.6;">{info['desc']}</p>
</div>
</div>
"""
                        st.markdown(html_str, unsafe_allow_html=True)
                else:
                    st.warning("⚠️ AI ค้นหาแถบสีไม่พบ กรุณาครอบรูปให้กระชับและเห็นแถบสีชัดเจนขึ้นครับ")
    elif uploaded_file:
        st.info("👈 กดปุ่ม '✨ ANALYZE NOW' เพื่อดูผลลัพธ์")
