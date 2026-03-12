import streamlit as st
from ultralytics import YOLO
from PIL import Image

# --- 1. หน้าจอแบบ Wide ---
st.set_page_config(page_title="Smart Urine Analyzer", page_icon="🧬", layout="wide")

# --- 2. CSS Compact Design ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;500;700&display=swap');
    html, body, [class*="css"] { font-family: 'Prompt', sans-serif; }
    #MainMenu, footer, header {visibility: hidden;}
    .block-container { padding-top: 1rem; }

    .result-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 10px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
        text-align: center;
        border: 1px solid #e2e8f0;
        margin-top: 10px;
        width: 180px; /* ล็อกขนาดการ์ดผลลัพธ์ให้เท่ากับรูป */
    }
    .badge {
        display: inline-block;
        padding: 5px 15px;
        border-radius: 15px;
        font-size: 16px;
        font-weight: 700;
        margin-bottom: 5px;
    }
    .neg { background: #ECFDF5; color: #059669; }
    .trace { background: #FFFBEB; color: #D97706; }
    .plus { background: #FEF2F2; color: #DC2626; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. Header ---
st.markdown("### 🧬 Smart Urine Analyzer <small style='font-size:14px; color:gray;'>| ระบบคัดกรองเบาหวานอัจฉริยะ</small>", unsafe_allow_html=True)
st.markdown("---")

# --- 4. Load Model (เพิ่มระบบแจ้งเตือนถ้าหาไฟล์ไม่เจอ) ---
@st.cache_resource
def load_model():
    try: 
        return YOLO('best (5).pt') 
    except Exception as e: 
        return str(e) # คืนค่า error ออกมาแจ้งเตือน

model = load_model()
class_names = ['Neg', 'Trace', 'plus1', 'plus2', 'plus3', 'plus4']

# --- 5. Main Layout (3 คอลัมน์) ---
col_upload, col_preview, col_result = st.columns([1.5, 1, 1.5])

with col_upload:
    st.markdown("**📸 1. อัปโหลดรูปภาพ**")
    
    # เช็คก่อนว่าโมเดลโหลดติดไหม ถ้าไม่ติดให้โชว์ Error สีแดง
    if isinstance(model, str):
        st.error(f"❌ โหลดโมเดลไม่ได้: โปรดเช็คว่าไฟล์ 'best (5).pt' อยู่ในโฟลเดอร์เดียวกับโค้ดไหม")
        uploaded_file = None
    else:
        uploaded_file = st.file_uploader("เลือกรูปแถบตรวจปัสสาวะ", type=["jpg", "jpeg", "png"])

# ระบบวิเคราะห์อัตโนมัติ (ไม่ต้องกดปุ่ม)
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    with col_preview:
        st.markdown("**🔍 2. รูปต้นฉบับ**")
        st.image(image, width=180) # ล็อกความกว้างรูปที่อัปโหลด
        
    with col_result:
        st.markdown("**🎯 3. ผลการวิเคราะห์**")
        with st.spinner('กำลังวิเคราะห์อัตโนมัติ...'):
            results = model(image)
            
            if len(results[0].boxes) > 0:
                box = results[0].boxes[0]
                label = class_names[int(box.cls[0])]
                conf = float(box.conf[0])
                b_style = "neg" if label == 'Neg' else "trace" if label == 'Trace' else "plus"
                
                # ล็อกความกว้างรูปที่ AI ตีกล่องแล้วให้เท่ากัน (180px)
                st.image(results[0].plot(), width=180) 
                
                st.markdown(f"""
                    <div class="result-card">
                        <div class="badge {b_style}">{label.upper()}</div>
                        <p style="margin:0; font-size:13px;">ความแม่นยำ: {conf:.1%}</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.error("⚠️ AI หาแถบตรวจไม่เจอ กรุณาถ่ายใหม่")
