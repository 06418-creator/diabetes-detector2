
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.set_page_config(page_title="Diabetes AI Detector", layout="centered")
st.title("🩺 Diabetes AI Analysis")

@st.cache_resource
def load_model():
    # ตรวจสอบว่ามีไฟล์ best.pt ไหม
    return YOLO('best.pt') 

model = load_model()

uploaded_file = st.file_uploader("เลือกรูปภาพแผ่นตรวจ...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='รูปภาพที่นำเข้า', use_column_width=True)
    
    with st.spinner('กำลังวิเคราะห์...'):
        results = model.predict(source=np.array(image), conf=0.434)
        if len(results[0].boxes) > 0:
            label = model.names[int(results[0].boxes[0].cls[0])]
            st.success(f"ระดับที่ตรวจพบ: {label}")
        else:
            st.error("ไม่พบแถบตรวจ กรุณาลองใหม่")
