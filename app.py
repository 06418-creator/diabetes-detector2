import streamlit as st
from ultralytics import YOLO
from PIL import Image

# --- 1. ตั้งค่าหน้าจอแบบ Wide ---
st.set_page_config(page_title="Smart Urine Analyzer", page_icon="🧬", layout="wide")

# --- 2. CSS สำหรับซ่อนส่วนเกินและตกแต่งการ์ด ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Prompt', sans-serif; }
    #MainMenu, footer, header {visibility: hidden;}
    .block-container { padding-top: 1rem; padding-bottom: 0rem; }

    /* ตกแต่งการ์ดผลลัพธ์ให้อ่านง่ายและสวยงาม */
    .info-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
        border-left: 6px solid #e2e8f0;
        margin-top: 0px;
    }
    
    .badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 18px;
        font-weight: 700;
        margin-bottom: 15px;
        color: white;
    }
    /* สีของแต่ละระดับ */
    .bg-neg { background-color: #10B981; } /* เขียว */
    .bg-trace { background-color: #F59E0B; } /* เหลือง/ส้ม */
    .bg-plus { background-color: #EF4444; } /* แดง */
    
    .section-title { font-weight: 700; color: #475569; font-size: 14px; margin-top: 10px; margin-bottom: 2px;}
    .section-detail { color: #1E293B; font-size: 16px; margin-bottom: 10px; line-height: 1.5;}
    </style>
    """, unsafe_allow_html=True)

# --- 3. Header ---
st.markdown("### 🧬 Smart Urine Analyzer <span style='font-size:16px; color:gray; font-weight:normal;'>| ระบบคัดกรองเบาหวานอัจฉริยะ</span>", unsafe_allow_html=True)
st.markdown("---")

# --- 4. โหลดโมเดล และ ฐานข้อมูลคำแนะนำ ---
@st.cache_resource
def load_model():
    try: return YOLO('best (5).pt') 
    except Exception as e: return str(e)

model = load_model()
class_names = ['Neg', 'Trace', 'plus1', 'plus2', 'plus3', 'plus4']

# ข้อมูลผลลัพธ์ ระดับอาการ และคำแนะนำ
class_info = {
    'Neg': {
        'title': 'NEGATIVE (ปกติ)', 'color': 'bg-neg',
        'level': 'ไม่พบน้ำตาลในปัสสาวะ (อยู่ในเกณฑ์ปกติ)',
        'action': 'ดูแลสุขภาพตามปกติ ทานอาหารที่มีประโยชน์ และออกกำลังกายสม่ำเสมอ'
    },
    'Trace': {
        'title': 'TRACE (พบเล็กน้อย)', 'color': 'bg-trace',
        'level': 'พบปริมาณน้ำตาลน้อยมาก (เริ่มมีความเสี่ยง)',
        'action': 'ควรเฝ้าระวัง ลดการทานของหวาน/น้ำอัดลม และควรตรวจซ้ำอีกครั้งใน 1-2 สัปดาห์'
    },
    'plus1': {
        'title': '+1 (ระดับเริ่มต้น)', 'color': 'bg-plus',
        'level': 'พบน้ำตาลประมาณ 100 mg/dL (ผิดปกติ)',
        'action': 'ปรับพฤติกรรมการกินด่วน! ลดแป้งและน้ำตาล หากมีอาการปัสสาวะบ่อย/หิวน้ำบ่อย ควรพบแพทย์'
    },
    'plus2': {
        'title': '+2 (ระดับปานกลาง)', 'color': 'bg-plus',
        'level': 'พบน้ำตาลประมาณ 250 mg/dL (เสี่ยงเบาหวาน)',
        'action': 'มีความเสี่ยงสูงที่จะเป็นโรคเบาหวาน แนะนำให้ไปโรงพยาบาลเพื่อเจาะเลือดตรวจอย่างละเอียด'
    },
    'plus3': {
        'title': '+3 (ระดับสูง)', 'color': 'bg-plus',
        'level': 'พบน้ำตาลประมาณ 500 mg/dL (อันตราย)',
        'action': 'ระดับน้ำตาลสูงมาก เสี่ยงต่อภาวะแทรกซ้อน <b>โปรดไปพบแพทย์เพื่อรับการรักษาทันที</b>'
    },
    'plus4': {
        'title': '+4 (ระดับสูงมาก)', 'color': 'bg-plus',
        'level': 'พบน้ำตาล > 1000 mg/dL (วิกฤต)',
        'action': '🚨 <b>อันตรายร้ายแรง!</b> อาจเกิดภาวะช็อกหรือเลือดเป็นกรด <b>โปรดไปโรงพยาบาลฉุกเฉินทันที</b>'
    }
}

# --- 5. Layout หลัก (แบ่งอัปโหลดไว้ด้านบน) ---
if isinstance(model, str):
    st.error(f"❌ โหลดโมเดลไม่ได้: ไม่พบไฟล์ 'best (5).pt'")
    uploaded_file = None
else:
    uploaded_file = st.file_uploader("📸 อัปโหลดรูปแผ่นตรวจปัสสาวะของคุณที่นี่ (คลิกหรือลากไฟล์มาวาง)", type=["jpg", "jpeg", "png"])

st.markdown("<br>", unsafe_allow_html=True)

# --- 6. ส่วนแสดงผล (ซ้าย: รูป / ขวา: คำแนะนำ) ---
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    with st.spinner('🤖 AI กำลังวิเคราะห์ผลลัพธ์...'):
        results = model(image)
        
        if len(results[0].boxes) > 0:
            box = results[0].boxes[0]
            cls_name = class_names[int(box.cls[0])]
            conf = float(box.conf[0])
            info = class_info[cls_name]
            
            # แบ่ง 2 คอลัมน์ (ซ้ายเล็ก ขวาใหญ่)
            col_img, col_text = st.columns([1, 2.5])
            
            with col_img:
                # โชว์รูปที่วิเคราะห์แล้ว (ล็อกความกว้างให้พอดี ไม่ใหญ่เกินไป)
                st.image(results[0].plot(), width=200)
                
            with col_text:
                # กล่องข้อความแสดงผลแบบละเอียด
                st.markdown(f"""
                    <div class="info-card" style="border-left-color: {'#10B981' if cls_name=='Neg' else '#F59E0B' if cls_name=='Trace' else '#EF4444'};">
                        <div class="badge {info['color']}">{info['title']}</div>
                        
                        <div class="section-title">📊 ความแม่นยำของ AI (Confidence)</div>
                        <div class="section-detail">{conf:.1%}</div>
                        
                        <div class="section-title">🩺 ระดับอาการ</div>
                        <div class="section-detail">{info['level']}</div>
                        
                        <div class="section-title">💡 สิ่งที่ควรทำ (คำแนะนำ)</div>
                        <div class="section-detail">{info['action']}</div>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.error("⚠️ AI หาแถบตรวจไม่เจอ กรุณาถ่ายรูปให้เห็นแถบสีชัดเจนขึ้นครับ")
