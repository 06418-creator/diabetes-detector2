import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os

# --- 1. ตั้งค่าหน้าจอแบบ Wide ---
st.set_page_config(page_title="Smart Urine Analyzer", page_icon="🧬", layout="wide")

# --- 2. CSS ตกแต่งให้กะทัดรัด ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;500;700&display=swap');
html, body, [class*="css"] { font-family: 'Prompt', sans-serif; }
#MainMenu, footer, header {visibility: hidden;}
.block-container { padding-top: 1rem; padding-bottom: 0rem; }
/* ปรับแต่งปุ่มให้อยู่กึ่งกลางพอดีกับรูป */
.stButton>button {
    width: 100%;
    height: 50px;
    border-radius: 8px;
    background-color: #2563EB;
    color: white;
    font-weight: bold;
    margin-top: 50px; 
}
</style>
""", unsafe_allow_html=True)

# --- 3. ส่วนหัว ---
col_h1, col_h2 = st.columns([1, 15])
with col_h1: st.markdown("## 🧬")
with col_h2: st.markdown("### Smart Urine Analyzer <small style='font-size:14px; color:gray;'>| ระบบคัดกรองเบาหวานเบื้องต้น</small>", unsafe_allow_html=True)
st.markdown("---")

# --- 4. โหลดโมเดล AI ---
# เช็คชื่อไฟล์ตรงนี้! ถ้าไฟล์คุณชื่ออื่น (เช่น best.pt) ให้เปลี่ยนข้อความในวงเล็บด้านล่างนี้
model_file = 'best (5).pt'

@st.cache_resource
def load_model():
    try: return YOLO(model_file) 
    except: return None

# ข้อมูลผลลัพธ์
class_names = ['Neg', 'Trace', 'plus1', 'plus2', 'plus3', 'plus4']
class_info = {
    'Neg': {'name': 'NEGATIVE (ปกติ)', 'info_type': 'info', 'desc': 'ไม่พบน้ำตาลในปัสสาวะ'},
    'Trace': {'name': 'TRACE (พบเล็กน้อย)', 'info_type': 'warning', 'desc': 'พบน้ำตาลปริมาณน้อยมาก ควรเฝ้าระวังและตรวจซ้ำ'},
    'plus1': {'name': '+1 (ระดับเริ่มต้น)', 'info_type': 'error', 'desc': 'พบน้ำตาล ~100 mg/dL ควรลดการทานของหวาน'},
    'plus2': {'name': '+2 (ระดับปานกลาง)', 'info_type': 'error', 'desc': 'พบน้ำตาล ~250 mg/dL ควรปรึกษาแพทย์เพื่อตรวจเลือด'},
    'plus3': {'name': '+3 (ระดับสูง)', 'info_type': 'error', 'desc': 'พบน้ำตาล ~500 mg/dL มีความเสี่ยงสูง โปรดพบแพทย์ทันที'},
    'plus4': {'name': '+4 (ระดับสูงมาก)', 'info_type': 'error', 'desc': 'พบน้ำตาล >1000 mg/dL 🚨 อันตรายร้ายแรง! โปรดพบแพทย์ทันที'}
}

# --- 5. Layout (ซ้าย: รูปและปุ่ม | ขวา: ผลลัพธ์) ---
col_left, col_right = st.columns([1.5, 2.5])

with col_left:
    st.markdown("**📸 1. จัดการรูปภาพ**")
    uploaded_file = st.file_uploader("เลือกรูปแถบตรวจปัสสาวะ", type=["jpg", "jpeg", "png"])
    
    analyze_now = False
    if uploaded_file:
        image = Image.open(uploaded_file)
        
        # แบ่งคอลัมน์ย่อยเพื่อให้ "รูป" อยู่ข้างๆ "ปุ่ม"
        col_img, col_btn = st.columns([1, 1])
        with col_img:
            # ล็อกขนาดรูปให้เล็ก ไม่ต้องเลื่อนจอ
            st.image(image, caption="รูปต้นฉบับ", width=160)
        with col_btn:
            analyze_now = st.button('✨ ANALYZE NOW')

with col_right:
    st.markdown("**🎯 2. ผลการวิเคราะห์**")
    
    # ดัก Error กรณีไม่มีไฟล์โมเดลในเครื่อง
    if not os.path.exists(model_file):
        st.error(f"❌ ระบบหาไฟล์โมเดล '{model_file}' ไม่พบครับ")
        st.info(f"**วิธีแก้ปัญหา:**\n1. ตรวจสอบว่ามีไฟล์ชื่อ `{model_file}` อยู่ในเครื่องหรือไม่\n2. นำไฟล์นั้นมาวางไว้ใน **โฟลเดอร์เดียวกัน** กับไฟล์โค้ดนี้\n3. หากไฟล์ของคุณชื่ออื่น (เช่น best.pt) ให้แก้ชื่อในโค้ดบรรทัดที่ 40 ครับ")
    
    elif uploaded_file and analyze_now:
        model = load_model()
        if model:
            with st.spinner('AI กำลังประมวลผล...'):
                results = model(image)
                
                if len(results[0].boxes) > 0:
                    box = results[0].boxes[0]
                    label = class_names[int(box.cls[0])]
                    conf = float(box.conf[0])
                    info = class_info[label]
                    
                    # รูป AI ตีกล่อง (ล็อกขนาดเท่ากัน)
                    st.image(results[0].plot(), caption="AI Detection Result", width=160)
                    st.divider()
                    
                    # แสดงผลแบบอ่านง่าย
                    if info['info_type'] == 'info':
                        st.info(f"📍 ผลลัพธ์: **{info['name']}**")
                    elif info['info_type'] == 'warning':
                        st.warning(f"📍 ผลลัพธ์: **{info['name']}**")
                    elif info['info_type'] == 'error':
                        st.error(f"📍 ผลลัพธ์: **{info['name']}**")

                    st.write(f"📊 ความแม่นยำของ AI (Confidence Level): **{conf:.1%}**")
                    st.write(f"💡 **คำแนะนำทางการแพทย์:** {info['desc']}")

                else:
                    st.error("⚠️ AI หาแถบตรวจไม่เจอ กรุณาครอบรูปให้พอดีแถบสีชัดเจนขึ้นครับ")
    elif uploaded_file:
        st.info("👈 กดปุ่ม 'ANALYZE NOW' ที่อยู่ข้างๆ รูปภาพเพื่อดูผล")
    else:
        st.info("👈 กรุณาอัปโหลดรูปภาพที่ฝั่งซ้าย")
