import streamlit as st
from ultralytics import YOLO
from PIL import Image

# --- 1. ตั้งค่าหน้าจอแบบ Wide และปรับ Layout ให้กะทัดรัด (Compact Design) ---
st.set_page_config(page_title="Smart Urine Analyzer", page_icon="🧬", layout="wide")

# --- 2. Custom CSS เพื่อลบระยะห่างและตกแต่งแอปให้ดูสะอาดตา ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;500;700&display=swap');
html, body, [class*="css"] { font-family: 'Prompt', sans-serif; }
#MainMenu, footer, header {visibility: hidden;}
.block-container { padding-top: 1rem; padding-bottom: 0rem; }
</style>
""", unsafe_allow_html=True)

# --- 3. ส่วนหัว (Header และ Title บรรทัดเดียว) ---
col_h1, col_h2 = st.columns([1, 8])
with col_h1: st.markdown("## 🧬")
with col_h2: st.markdown("### Smart Urine Analyzer <small style='font-size:14px; color:gray;'>| ระบบคัดกรองเบาหวานเบื้องต้นอัจฉริยะ</small>", unsafe_allow_html=True)
st.markdown("---")

# --- 4. โหลดโมเดล AI ---
@st.cache_resource
def load_model():
    try: return YOLO('best (5).pt') 
    except: return None
model = load_model()

# กำหนดข้อมูลคลาสภาษาไทยให้เรียบร้อย
class_names = ['Neg', 'Trace', 'plus1', 'plus2', 'plus3', 'plus4']
class_info = {
    'Neg': {'name': 'NEGATIVE (ปกติ)', 'info_type': 'info', 'desc': 'ไม่พบน้ำตาลในปัสสาวะ'},
    'Trace': {'name': 'TRACE (พบเล็กน้อย)', 'info_type': 'warning', 'desc': 'พบน้ำตาลปริมาณน้อยมาก ควรเฝ้าระวัง'},
    'plus1': {'name': '+1 (ระดับเริ่มต้น)', 'info_type': 'error', 'desc': 'พบน้ำตาล ~100 mg/dL ควรปรับพฤติกรรมการกิน'},
    'plus2': {'name': '+2 (ระดับปานกลาง)', 'info_type': 'error', 'desc': 'พบน้ำตาล ~250 mg/dL ควรปรึกษาแพทย์'},
    'plus3': {'name': '+3 (ระดับสูง)', 'info_type': 'error', 'desc': 'พบน้ำตาล ~500 mg/dL มีความเสี่ยงสูง ควรพบแพทย์ทันที'},
    'plus4': {'name': '+4 (ระดับสูงมาก)', 'info_type': 'error', 'desc': 'พบน้ำตาล >1000 mg/dL 🚨 อันตรายร้ายแรง! ควรพบแพทย์ทันที'}
}

# --- 5. Main Layout แบ่ง 2 คอลัมน์แบบ [กะทัดรัด, ผลลัพธ์สมบูรณ์] ---
col_input, col_result = st.columns([1.5, 2.5])

with col_input:
    st.markdown("**📸 1. จัดการรูปภาพ**")
    # File Uploader
    uploaded_file = st.file_uploader("เลือกรูปแถบตรวจปัสสาวะ", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        # ✅ แก้ไขจุดที่ 2: บังคับย่อรูปต้นฉบับให้เล็กพอดี ( width=180) เพื่อไม่ให้ล้นหน้าจอ
        st.image(image, caption="รูปที่อัปโหลด", width=180)
        
        # ✅ แก้ไขจุดที่ 1: นำปุ่ม Analyze กลับมา
        analyze_now = st.button('✨ ANALYZE NOW')
    else:
        analyze_now = False

with col_result:
    st.markdown("**🎯 2. ผลการวิเคราะห์**")
    
    if uploaded_file and analyze_now:
        if model:
            with st.spinner('AI กำลังประมวลผล...'):
                results = model(image)
                
                if len(results[0].boxes) > 0:
                    box = results[0].boxes[0]
                    label = class_names[int(box.cls[0])]
                    conf = float(box.conf[0])
                    info = class_info[label]
                    
                    # ✅ แก้ไขจุดที่ 2: ย่อรูปผลลัพธ์ที่ AI ตีกล่องแล้วให้เล็กพอดี ( width=180)
                    st.image(results[0].plot(), caption="AI Detection Result", width=180)
                    
                    # ✅ แก้ไขจุดที่ 3: แก้ไขคำอธิบายที่ "ชุ่ย" ใน `image_3.png` โดยใช้ฟังก์ชันnativeอ่านง่าย
                    st.divider()
                    
                    # เลือกใช้กล่องสถานะตามระดับผลลัพธ์ (เขียว/เหลือง/แดง)
                    if info['info_type'] == 'info':
                        st.info(f"📍 ผลลัพธ์: {info['name']}")
                    elif info['info_type'] == 'warning':
                        st.warning(f"📍 ผลลัพธ์: {info['name']}")
                    elif info['info_type'] == 'error':
                        st.error(f"📍 ผลลัพธ์: {info['name']}")

                    st.write(f"📊 ความแม่นยำของ AI (Confidence Level): **{conf:.1%}**")
                    st.divider()
                    
                    st.subheader("💡 คำแนะนำทางการแพทย์เบื้องต้น:")
                    st.write(f"  > **{info['desc']}**")
                    st.markdown("Disclaimer: This tool provides AI-based estimations. Always consult a healthcare professional for accurate medical diagnosis.")

                else:
                    st.error("⚠️ AI หาแถบตรวจไม่เจอ กรุณาครอบรูปให้พอดีแถบสีชัดเจนขึ้นครับ")
        else:
            st.error("❌ ไม่พบไฟล์โมเดล AI (best (5).pt)")
            
    elif uploaded_file:
        st.info("👆 ตรวจสอบรูปภาพทางซ้าย แล้วกดปุ่ม 'ANALYZE NOW' ทางซ้ายเพื่อดูผล")
    else:
        st.info("👈 กรุณาอัปโหลดรูปภาพที่ฝั่งซ้าย")
