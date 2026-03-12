import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os

# --- 1. ตั้งค่าหน้าจอแบบ Wide ---
st.set_page_config(page_title="Smart Urine Analyzer", page_icon="🧬", layout="wide")

# --- 2. CSS สำหรับตกแต่งแอปให้สวยงาม (ซ่อนส่วนเกิน จัดการ์ดผลลัพธ์) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Prompt', sans-serif; }
    #MainMenu, footer, header {visibility: hidden;}
    .block-container { padding-top: 2rem; padding-bottom: 0rem; }

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
    
    .bg-neg { background-color: #10B981; } /* เขียว */
    .bg-trace { background-color: #F59E0B; } /* เหลือง/ส้ม */
    .bg-plus { background-color: #EF4444; } /* แดง */
    
    .section-title { font-weight: 700; color: #475569; font-size: 14px; margin-top: 10px; margin-bottom: 2px;}
    .section-detail { color: #1E293B; font-size: 16px; margin-bottom: 10px; line-height: 1.5;}
    </style>
    """, unsafe_allow_html=True)

# --- 3. ส่วนหัว (Header) ---
st.markdown("### 🧬 Smart Urine Analyzer <span style='font-size:16px; color:gray; font-weight:normal;'>| ระบบคัดกรองเบาหวานอัจฉริยะ</span>", unsafe_allow_html=True)
st.markdown("---")

# --- 4. ฐานข้อมูลคำแนะนำ (อ้างอิงตามระดับน้ำตาล) ---
class_names = ['Neg', 'Trace', 'plus1', 'plus2', 'plus3', 'plus4']
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

# --- 5. โหลดโมเดล AI ---
# เช็คชื่อไฟล์ตรงนี้! ถ้าไฟล์คุณชื่ออื่น ให้แก้ตรง 'best (5).pt'
model_path = 'best (5).pt' 

@st.cache_resource
def load_model(path):
    if not os.path.exists(path):
        return f"ไม่พบไฟล์โมเดล '{path}' ในโฟลเดอร์"
    try: 
        return YOLO(path) 
    except Exception as e: 
        return f"โหลดโมเดลผิดพลาด: {e}"

model = load_model(model_path)

# --- 6. ส่วนอัปโหลดและแสดงผล ---
if isinstance(model, str):
    # ถ้าโหลดโมเดลไม่ผ่าน จะโชว์ Error สีแดง
    st.error(f"❌ {model}")
    st.info("💡 วิธีแก้: กรุณาตรวจสอบว่ามีไฟล์ชื่อ 'best (5).pt' วางอยู่ในโฟลเดอร์เดียวกันกับไฟล์โค้ดนี้หรือไม่ (ถ้าชื่อไฟล์ต่างกัน ให้แก้ชื่อในบรรทัดที่ 67)")
else:
    # อัปโหลดรูปภาพ
    uploaded_file = st.file_uploader("📸 อัปโหลดรูปแผ่นตรวจปัสสาวะของคุณที่นี่ (รอสักครู่ระบบจะวิเคราะห์อัตโนมัติ)", type=["jpg", "jpeg", "png"])
    st.markdown("<br>", unsafe_allow_html=True)

    # เมื่อมีการอัปโหลดรูป
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        with st.spinner('🤖 AI กำลังวิเคราะห์ผลลัพธ์...'):
            results = model(image)
            
            if len(results[0].boxes) > 0:
                box = results[0].boxes[0]
                cls_name = class_names[int(box.cls[0])]
                conf = float(box.conf[0])
                info = class_info[cls_name]
                
                # กำหนดสีขอบซ้ายของการ์ดตามระดับความรุนแรง
                border_color = '#10B981' if cls_name == 'Neg' else '#F59E0B' if cls_name == 'Trace' else '#EF4444'
                
                # แบ่งหน้าจอเป็น 2 ฝั่ง (ซ้าย: รูป / ขวา: ผลลัพธ์)
                col_img, col_text = st.columns([1, 2.5])
                
                with col_img:
                    # แสดงรูปที่ AI วิเคราะห์แล้ว พร้อมล็อกขนาดความกว้างที่ 250px เพื่อไม่ให้ใหญ่ล้นจอ
                    st.image(results[0].plot(), width=250)
                    
                with col_text:
                    # แสดงกล่องข้อความผลลัพธ์
                    st.markdown(f"""
                        <div class="info-card" style="border-left-color: {border_color};">
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
                st.error("⚠️ AI หาแถบตรวจไม่เจอ กรุณาถ่ายรูปให้เห็นแถบสีชัดเจนขึ้น หรือครอบรูปให้พอดีแถบตรวจครับ")
