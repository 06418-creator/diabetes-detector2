import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# 1. การตั้งค่าหน้าตาเว็บเบื้องต้น
st.set_page_config(page_title="Diabetes AI Detector", layout="wide")

# 2. ใส่ CSS เพื่อความสวยงาม
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #007bff;
        color: white;
    }
    .result-card {
        padding: 20px;
        border-radius: 15px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🩺 Diabetes AI Analysis System")
st.write("ระบบวิเคราะห์ระดับน้ำตาลจากแผ่นตรวจด้วยเทคโนโลยี AI")

# โหลดโมเดล
@st.cache_resource
def load_model():
    return YOLO('best.pt') 

model = load_model()

# 3. จัด Layout เป็น 2 คอลัมน์ (ซ้ายอัปโหลด - ขวาแสดงผล)
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 อัปโหลดรูปภาพ")
    uploaded_file = st.file_uploader("เลือกรูปภาพแผ่นตรวจของคุณ...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='รูปภาพที่นำเข้า', use_column_width=True)

with col2:
    st.subheader("📊 ผลการวิเคราะห์")
    if uploaded_file is not None:
        with st.spinner('AI กำลังตรวจสอบ...'):
            results = model.predict(source=np.array(image), conf=0.434)
            
            if len(results[0].boxes) > 0:
                # ดึงค่าผลลัพธ์
                box = results[0].boxes[0]
                label = model.names[int(box.cls[0])]
                conf = box.conf[0]
                
                # แสดงผลแบบสวยๆ
                st.markdown(f"""
                <div class="result-card">
                    <h2 style='color: #28a745;'>ตรวจพบระดับ: {label}</h2>
                    <p style='font-size: 18px;'>ค่าความเชื่อมั่น (Confidence): {conf:.2%}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # เพิ่มคำแนะนำตามระดับที่เจอ
                if "positive" in label.lower():
                    st.warning("⚠️ คำแนะนำ: ผลตรวจมีความเสี่ยง โปรดปรึกษาแพทย์ผู้เชี่ยวชาญ")
                else:
                    st.success("✅ คำแนะนำ: ระดับน้ำตาลอยู่ในเกณฑ์ปกติ")
            else:
                st.error("❌ ไม่พบแถบตรวจในรูปภาพ กรุณาถ่ายรูปให้ชัดเจนและลองใหม่")
    else:
        st.info("กรุณาอัปโหลดรูปภาพที่ด้านซ้ายเพื่อเริ่มการวิเคราะห์")
