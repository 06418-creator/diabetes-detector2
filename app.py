import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

st.set_page_config(page_title="Diabetes AI Detector", layout="wide")

@st.cache_resource
def load_model():
    return YOLO('best.pt') 

model = load_model()

st.title("🩺 Diabetes AI Debugger")
uploaded_file = st.file_uploader("ส่งรูปมาพิสูจน์กัน...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    # ปรับค่า Confidence ตรงนี้ได้ (ลองลดเหลือ 0.25)
    conf_threshold = st.sidebar.slider("ปรับความละเอียด AI (Confidence)", 0.0, 1.0, 0.25)
    
    results = model.predict(source=img_array, conf=conf_threshold)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🖼️ รูปต้นฉบับ")
        st.image(image, use_column_width=True)
        
    with col2:
        st.subheader("🔍 ผลที่ AI มองเห็น")
        # สั่งให้ YOLO วาดกรอบลงบนภาพเลย (Plotting)
        res_plotted = results[0].plot()
        # แปลงสี BGR เป็น RGB (เพราะ OpenCV กับ PIL สลับสีกัน)
        res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
        st.image(res_rgb, caption='AI Detect Results', use_column_width=True)

        if len(results[0].boxes) > 0:
            for box in results[0].boxes:
                label = model.names[int(box.cls[0])]
                score = box.conf[0]
                st.success(f"พบ: {label} (ความมั่นใจ {score:.2%})")
        else:
            st.error("AI ยังหาไม่เจอ... ลองปรับ Confidence ลดลงหรือถ่ายรูปใหม่ให้ชัดขึ้นครับ")
