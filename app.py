import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageOps
import numpy as np
import cv2

st.set_page_config(page_title="Diabetes AI Detector", layout="wide")

@st.cache_resource
def load_model():
    # โหลดโมเดลและตั้งค่าโหมดประเมินผล
    model = YOLO('best.pt')
    return model

model = load_model()

st.title("🩺 Diabetes AI Analysis")

uploaded_file = st.file_uploader("ส่งรูปภาพที่ใช้เทรนหรือรูปใหม่มาตรวจสอบ...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. อ่านภาพและจัดการเรื่องการหมุนภาพอัตโนมัติ (Exif orientation)
    image = Image.open(uploaded_file)
    image = ImageOps.exif_transpose(image) 
    
    # 2. แปลงเป็น Numpy Array และทำให้สีเป็น RGB แน่นอน
    img_array = np.array(image.convert('RGB'))
    
    # 3. สั่งทำนายผล (ลองลด conf ลงชั่วคราวเพื่อดูว่ามันเจออะไรบ้างไหม)
    # เพิ่ม imgsz=640 เพื่อให้ขนาดภาพเท่ากับตอนเทรนส่วนใหญ่
    results = model.predict(source=img_array, conf=0.20, imgsz=640)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🖼️ รูปภาพที่คุณส่งมา")
        st.image(image, use_container_width=True)
        
    with col2:
        st.subheader("🔍 ผลการสแกนจาก AI")
        
        # วาดผลลัพธ์
        res_plotted = results[0].plot(conf=True, labels=True)
        st.image(res_plotted, caption='AI Detection Result', use_container_width=True)

        if len(results[0].boxes) > 0:
            for box in results[0].boxes:
                label = model.names[int(box.cls[0])]
                score = box.conf[0]
                st.success(f"✅ ตรวจพบ: {label} (ความมั่นใจ {score:.2%})")
        else:
            st.error("⚠️ AI ยังหาแถบตรวจไม่เจอ")
            st.info("คำแนะนำ: ลองถ่ายรูปให้ใกล้ขึ้น หรือให้แถบตรวจอยู่กลางภาพ และมีแสงสว่างที่เพียงพอ")
