import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# --- การตั้งค่าหน้าจอ ---
st.set_page_config(page_title="Diabetes Strip Analyzer", layout="centered")

# --- สไตล์ CSS เพื่อความสวยงาม ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 10px; height: 3em; background-color: #4CAF50; color: white; }
    .result-box { padding: 20px; border-radius: 15px; text-align: center; font-size: 24px; font-weight: bold; margin-top: 20px; }
    .neg { background-color: #e8f5e9; color: #2e7d32; border: 2px solid #2e7d32; }
    .trace { background-color: #fff3e0; color: #ef6c00; border: 2px solid #ef6c00; }
    .plus { background-color: #ffebee; color: #c62828; border: 2px solid #c62828; }
    </style>
    """, unsafe_allow_status_code=True)

st.title("🧪 ระบบตรวจวิเคราะห์แถบสีปัสสาวะ")
st.write("อัปโหลดรูปภาพแผ่นตรวจเพื่อวิเคราะห์ระดับน้ำตาล (รองรับ 6 ระดับ)")

# --- โหลดโมเดล (ต้องชื่อ best.pt ตามที่อัปโหลดขึ้น GitHub) ---
@st.cache_resource
def load_model():
    return YOLO('best.pt')

model = load_model()

# --- รายชื่อคลาสและสีที่เกี่ยวข้อง ---
class_info = {
    'Neg': {'name': 'ปกติ (Negative)', 'class': 'neg', 'desc': 'ไม่พบน้ำตาลในปัสสาวะ'},
    'Trace': {'name': 'เล็กน้อย (Trace)', 'class': 'trace', 'desc': 'พบน้ำตาลในปริมาณน้อยมาก'},
    'plus1': {'name': 'ระดับ 1 (+1)', 'class': 'plus', 'desc': 'พบน้ำตาลในระดับเริ่มต้น'},
    'plus2': {'name': 'ระดับ 2 (+2)', 'class': 'plus', 'desc': 'พบน้ำตาลในระดับปานกลาง'},
    'plus3': {'name': 'ระดับ 3 (+3)', 'class': 'plus', 'desc': 'พบน้ำตาลในระดับสูง'},
    'plus4': {'name': 'ระดับ 4 (+4)', 'class': 'plus', 'desc': 'พบน้ำตาลในระดับสูงมาก ควรพบแพทย์'}
}
class_names = ['Neg', 'Trace', 'plus1', 'plus2', 'plus3', 'plus4']

# --- ส่วนการอัปโหลดไฟล์ ---
uploaded_file = st.file_uploader("เลือกรูปภาพแผ่นตรวจ...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='รูปภาพที่อัปโหลด', use_container_width=True)
    
    if st.button('เริ่มการวิเคราะห์'):
        with st.spinner('AI กำลังวิเคราะห์สี...'):
            # ทำการ Prediction
            results = model(image)
            
            if len(results[0].boxes) > 0:
                # ดึงคลาสที่ AI มั่นใจที่สุด
                box = results[0].boxes[0]
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                
                label = class_names[cls_id]
                info = class_info[label]
                
                # แสดงรูปที่ตีกรอบแล้ว
                res_plotted = results[0].plot()
                st.image(res_plotted, caption='ผลการตรวจจับตำแหน่ง', use_container_width=True)
                
                # แสดงกล่องผลลัพธ์สวยๆ
                st.markdown(f"""
                    <div class="result-box {info['class']}">
                        ผลลัพธ์: {info['name']}<br>
                        <span style="font-size: 16px;">ความแม่นยำ: {conf:.2%}</span>
                    </div>
                    <div style="text-align: center; margin-top: 10px; color: #666;">
                        คำแนะนำ: {info['desc']}
                    </div>
                """, unsafe_allow_status_code=True)
                
            else:
                st.error("❌ ไม่สามารถตรวจพบแถบสีในรูปภาพได้ กรุณาถ่ายรูปให้ชัดเจนขึ้น")

st.info("💡 หมายเหตุ: ผลการวิเคราะห์นี้เป็นการประมาณการโดย AI โปรดปรึกษาแพทย์เพื่อยืนยันผล")
