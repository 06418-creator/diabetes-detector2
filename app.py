import streamlit as st
from ultralytics import YOLO
from PIL import Image

# 1. ตั้งค่าหน้าจอ
st.set_page_config(page_title="Diabetes Strip Analyzer", layout="centered")

# 2. CSS Style
st.markdown("""
    <style>
    .stButton>button { width: 100%; border-radius: 10px; background-color: #4CAF50; color: white; padding: 10px; font-size: 18px; font-weight: bold;}
    .result-box { padding: 20px; border-radius: 15px; text-align: center; font-size: 24px; font-weight: bold; margin-bottom: 20px;}
    .neg { background-color: #e8f5e9; color: #2e7d32; border: 2px solid #2e7d32; }
    .trace { background-color: #fff3e0; color: #ef6c00; border: 2px solid #ef6c00; }
    .plus { background-color: #ffebee; color: #c62828; border: 2px solid #c62828; }
    </style>
    """, unsafe_allow_html=True)

st.title("🧪 วิเคราะห์แถบสีปัสสาวะ")
st.write("อัปโหลดรูปภาพแผ่นตรวจเพื่อวิเคราะห์ระดับน้ำตาล (6 ระดับ)")

# 3. โหลดโมเดล
@st.cache_resource
def load_model():
    try:
        return YOLO('best (4).pt')
    except Exception as e:
        st.error(f"ไม่สามารถโหลดโมเดลได้: {e}")
        return None

model = load_model()

# 4. ข้อมูลคลาส
class_info = {
    'Neg': {'name': 'ปกติ (Negative)', 'class': 'neg', 'desc': 'ไม่พบน้ำตาลในปัสสาวะ'},
    'Trace': {'name': 'เล็กน้อย (Trace)', 'class': 'trace', 'desc': 'พบน้ำตาลในปริมาณน้อยมาก'},
    'plus1': {'name': 'ระดับ 1 (+1)', 'class': 'plus', 'desc': 'พบน้ำตาลในระดับเริ่มต้น (100 mg/dL)'},
    'plus2': {'name': 'ระดับ 2 (+2)', 'class': 'plus', 'desc': 'พบน้ำตาลในระดับปานกลาง (250 mg/dL)'},
    'plus3': {'name': 'ระดับ 3 (+3)', 'class': 'plus', 'desc': 'พบน้ำตาลในระดับสูง (500 mg/dL)'},
    'plus4': {'name': 'ระดับ 4 (+4)', 'class': 'plus', 'desc': 'พบน้ำตาลในระดับสูงมาก (>1000 mg/dL) ควรพบแพทย์ทันที'}
}
class_names = ['Neg', 'Trace', 'plus1', 'plus2', 'plus3', 'plus4']

# 5. การอัปโหลดและวิเคราะห์
uploaded_file = st.file_uploader("เลือกรูปภาพแผ่นตรวจ...", type=["jpg", "jpeg", "png"])

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
                    
                    st.image(results[0].plot(), caption='ผลการตรวจจับ', use_container_width=True)
                    
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
                    st.warning("⚠️ ไม่พบแถบตรวจในรูปภาพ กรุณาลองถ่ายรูปใหม่ให้ชัดเจนและใกล้ขึ้น")
        else:
            st.error("โมเดลไม่พร้อมใช้งาน")

st.markdown("---")
st.info("💡 หมายเหตุ: ผลการวิเคราะห์นี้เป็นการประมาณการโดย AI โปรดปรึกษาแพทย์เพื่อยืนยันผลทางการแพทย์")
