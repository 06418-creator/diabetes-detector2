import streamlit as st
from ultralytics import YOLO
from PIL import Image

# 1. ตั้งค่าหน้าจอ
st.set_page_config(page_title="Diabetes Strip Analyzer", layout="centered")

# 2. CSS Style
st.markdown("""
    <style>
    .stButton>button { width: 100%; border-radius: 10px; background-color: #4CAF50; color: white; }
    .result-box { padding: 20px; border-radius: 15px; text-align: center; font-size: 24px; font-weight: bold; }
    .neg { background-color: #e8f5e9; color: #2e7d32; border: 2px solid #2e7d32; }
    .trace { background-color: #fff3e0; color: #ef6c00; border: 2px solid #ef6c00; }
    .plus { background-color: #ffebee; color: #c62828; border: 2px solid #c62828; }
    </style>
    """, unsafe_allow_html=True)

st.title("🧪 วิเคราะห์แถบสีปัสสาวะ")

# 3. โหลดโมเดล
@st.cache_resource
def load_model():
    try:
        return YOLO('best (4).pt')
    except:
        return None

model = load_model()

# 4. ข้อมูลคลาส
class_info = {
    'Neg': {'n': 'ปกติ (Negative)', 'c': 'neg', 'd': 'ไม่พบน้ำตาลในปัสสาวะ'},
    'Trace': {'n': 'เล็กน้อย (Trace)', 'c': 'trace', 'd': 'พบน้ำตาลในปริมาณน้อยมาก'},
    'plus1': {'n': 'ระดับ 1 (+1)', 'c': 'plus', 'd': 'พบน้ำตาลระดับเริ่มต้น'},
    'plus2': {'n': 'ระดับ 2 (+2)', 'c': 'plus', 'd': 'พบน้ำตาลระดับปานกลาง'},
    'plus3': {'n': 'ระดับ 3 (+3)', 'c': 'plus', 'd': 'พบน้ำตาลระดับสูง'},
    'plus4': {'n': 'ระดับ 4 (+4)', 'c': 'plus', 'd': 'พบน้ำตาลระดับสูงมาก'}
}
class_names = ['Neg', 'Trace', 'plus1', 'plus2', 'plus3', 'plus4']

# 5. การอัปโหลดและวิเคราะห์
up_file = st.file_uploader("เลือกรูปภาพ...", type=["jpg", "png", "jpeg"])

if up_file:
    img = Image.open(up_file)
    st.image(img, caption='รูปที่อัปโหลด', use_container_width=True)
    
    if st.button('🚀 เริ่มวิเคราะห์'):
        if model:
            results = model(img)
            if len(results[0].boxes) > 0:
                box = results[0].boxes[0]
                idx = int(box.cls[0])
                conf = float(box.conf[0])
                
                info = class_info[class_names[idx]]
                
                st.image(results[0].plot(), use_container_width=True)
                
                html = f"""
                <div class="result-box {info['c']}">
                    {info['n']}<br>
                    <span style="font-size:14px">มั่นใจ: {conf:.1%}</span>
                </div>
                <p style="text-align:center; margin-top:10px"><b>คำแนะนำ:</b> {info['d']}</p>
                """
                st.markdown(html, unsafe_allow_html=True)
            else:
                st.warning("ไม่พบแถบตรวจ")
        else:
            st.error("ไม่พบโมเดล best (4).pt")

st.info("หมายเหตุ: ผลลัพธ์เป็นการประมาณการโดย AI")    'Neg': {'name': 'ปกติ (Negative)', 'class': 'neg', 'desc': 'ไม่พบน้ำตาลในปัสสาวะ'},
    'Trace': {'name': 'เล็กน้อย (Trace)', 'class': 'trace', 'desc': 'พบน้ำตาลในปริมาณน้อยมาก'},
    'plus1': {'name': 'ระดับ 1 (+1)', 'class': 'plus', 'desc': 'พบน้ำตาลในระดับเริ่มต้น (100 mg/dL)'},
    'plus2': {'name': 'ระดับ 2 (+2)', 'class': 'plus', 'desc': 'พบน้ำตาลในระดับปานกลาง (250 mg/dL)'},
    'plus3': {'name': 'ระดับ 3 (+3)', 'class': 'plus', 'desc': 'พบน้ำตาลในระดับสูง (500 mg/dL)'},
    'plus4': {'name': 'ระดับ 4 (+4)', 'class': 'plus', 'desc': 'พบน้ำตาลในระดับสูงมาก (>1000 mg/dL)'}
}
class_names = ['Neg', 'Trace', 'plus1', 'plus2', 'plus3', 'plus4']

# --- 5. การอัปโหลดและวิเคราะห์ ---
uploaded_file = st.file_uploader("เลือกรูปภาพ...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='รูปภาพที่อัปโหลด', use_container_width=True)
    
    if st.button('🚀 เริ่มการวิเคราะห์'):
        if model is not None:
            with st.spinner('กำลังวิเคราะห์...'):
                results = model(image)
                
                if len(results[0].boxes) > 0:
                    top_box = results[0].boxes[0]
                    cls_id = int(top_box.cls[0])
                    conf = float(top_box.conf[0])
                    
                    label = class_names[cls_id]
                    info = class_info[label]
                    
                    # แสดงรูป
                    st.image(results[0].plot(), caption='ผลการตรวจจับ', use_container_width=True)
                    
                    # แสดงผลลัพธ์ (แก้จุดที่มักเกิด Syntax Error ใน f-string)
                    res_html = f"""
                        <div class="result-box {info['class']}">
                            ผลการวิเคราะห์: {info['name']}<br>
                            <span style="font-size: 16px;">ความเชื่อมั่น: {conf:.2%}</span>
                        </div>
                        <div style="text-align: center; margin-top: 15px; color: #333;">
                            <b>คำแนะนำ:</b> {info['desc']}
                        </div>
                    """
                    st.markdown(res_html, unsafe_allow_html=True)
                else:
                    st.warning("⚠️ ไม่พบแถบตรวจในรูปภาพ")
        else:
            st.error("โมเดลไม่พร้อมใช้งาน")

st.markdown("---")
st.info("💡 หมายเหตุ: ผลวิเคราะห์เป็นเพียงการประมาณการเบื้องต้น")class_info = {
    'Neg': {'name': 'ปกติ (Negative)', 'class': 'neg', 'desc': 'ไม่พบน้ำตาลในปัสสาวะ'},
    'Trace': {'name': 'เล็กน้อย (Trace)', 'class': 'trace', 'desc': 'พบน้ำตาลในปริมาณน้อยมาก'},
    'plus1': {'name': 'ระดับ 1 (+1)', 'class': 'plus', 'desc': 'พบน้ำตาลในระดับเริ่มต้น (100 mg/dL)'},
    'plus2': {'name': 'ระดับ 2 (+2)', 'class': 'plus', 'desc': 'พบน้ำตาลในระดับปานกลาง (250 mg/dL)'},
    'plus3': {'name': 'ระดับ 3 (+3)', 'class': 'plus', 'desc': 'พบน้ำตาลในระดับสูง (500 mg/dL)'},
    'plus4': {'name': 'ระดับ 4 (+4)', 'class': 'plus', 'desc': 'พบน้ำตาลในระดับสูงมาก (>1000 mg/dL) ควรพบแพทย์ทันที'}
}
class_names = ['Neg', 'Trace', 'plus1', 'plus2', 'plus3', 'plus4']

# --- 5. ส่วนการอัปโหลดและวิเคราะห์ ---
uploaded_file = st.file_uploader("เลือกรูปภาพแผ่นตรวจ...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='รูปภาพที่อัปโหลด', use_container_width=True)
    
    if st.button('🚀 เริ่มการวิเคราะห์'):
        if model is not None:
            with st.spinner('AI กำลังตรวจสอบความเข้มสี...'):
                results = model(image)
                
                if len(results[0].boxes) > 0:
                    # เลือกกล่องที่มีค่าความมั่นใจสูงสุด
                    top_box = results[0].boxes[0]
                    cls_id = int(top_box.cls[0])
                    conf = float(top_box.conf[0])
                    
                    label = class_names[cls_id]
                    info = class_info[label]
                    
                    # แสดงรูปที่ตีกรอบ
                    res_plotted = results[0].plot()
                    st.image(res_plotted, caption='ตำแหน่งที่ AI ตรวจพบ', use_container_width=True)
                    
                    # แสดงผลลัพธ์ UI สวยๆ
                    st.markdown(f"""
                        <div class="result-box {info['class']}">
                            ผลการวิเคราะห์: {info['name']}<br>
                            <span style="font-size: 16px;">ความเชื่อมั่นของ AI: {conf:.2%}</span>
                        </div>
                        <div style="text-align: center; margin-top: 15px; color: #333; font-size: 18px; padding: 10px; border-left: 5px solid #ccc;">
                            <b>คำแนะนำ:</b> {info['desc']}
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning("⚠️ ไม่พบแถบตรวจในรูปภาพ กรุณาถ่ายรูปให้เห็นช่องสีชัดเจนและไม่มีเงาบัง")
        else:
            st.error("โมเดลไม่พร้อมใช้งาน")

st.markdown("---")
st.info("💡 หมายเหตุ: ระบบนี้เป็นเพียงเครื่องมือช่วยวิเคราะห์เบื้องต้น โปรดปรึกษาบุคลากรทางการแพทย์เพื่อผลลัพธ์ที่ถูกต้องที่สุด")    'Neg': {'name': 'ปกติ (Negative)', 'class': 'neg', 'desc': 'ไม่พบน้ำตาลในปัสสาวะ'},
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
        if model is not None:
            with st.spinner('AI กำลังวิเคราะห์สี...'):
                results = model(image)
                
                # ตรวจสอบว่าเจอ Object หรือไม่
                if len(results[0].boxes) > 0:
                    # เลือกอันที่มีความมั่นใจสูงที่สุด
                    top_box = results[0].boxes[0]
                    cls_id = int(top_box.cls[0])
                    conf = float(top_box.conf[0])
                    
                    label = class_names[cls_id]
                    info = class_info[label]
                    
                    # แสดงรูปภาพผลลัพธ์
                    res_plotted = results[0].plot()
                    st.image(res_plotted, caption='ตำแหน่งที่ตรวจพบ', use_container_width=True)
                    
                    # แสดงผลลัพธ์ UI
                    st.markdown(f"""
                        <div class="result-box {info['class']}">
                            ผลลัพธ์: {info['name']}<br>
                            <span style="font-size: 16px;">ความแม่นยำ: {conf:.2%}</span>
                        </div>
                        <div style="text-align: center; margin-top: 15px; color: #333; font-size: 18px;">
                            <b>คำแนะนำ:</b> {info['desc']}
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning("⚠️ ไม่พบแถบตรวจในรูปภาพ กรุณาลองถ่ายรูปใหม่ให้ชัดเจนและใกล้ขึ้น")
        else:
            st.error("ไม่พบโมเดล best.pt ในระบบ")

st.markdown("---")
st.info("💡 หมายเหตุ: ผลการวิเคราะห์นี้เป็นการประมาณการโดย AI โปรดปรึกษาแพทย์เพื่อยืนยันผลทางการแพทย์")
