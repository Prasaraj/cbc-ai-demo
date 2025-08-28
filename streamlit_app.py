import streamlit as st
import pandas as pd
import requests

# --- ตั้งค่าหน้าเว็บ ---
st.set_page_config(page_title="AI Blood Test Analyzer", page_icon="🩸", layout="wide")
st.title("🩸 Hybrid AI System for CBC Analysis")
st.write("ระบบผู้ช่วยวิเคราะห์ผลตรวจความสมบูรณ์ของเม็ดเลือด (CBC) เบื้องต้น")

# --- สร้าง User Interface ---
st.header("กรอกข้อมูลผลเลือดของคนไข้")
col1, col2, col3 = st.columns(3)
with col1:
    age_y = st.number_input("อายุ (ปี)", 0, 120, 35, 1)
    sex = st.selectbox("เพศ", ("Female", "Male"))
    hct = st.number_input("HCT (%)", 10.0, 70.0, 38.0, 0.1)
with col2:
    mcv = st.number_input("MCV (fL)", 50.0, 150.0, 85.0, 0.1)
    wbc = st.number_input("WBC (cells/mcL)", 1000.0, 50000.0, 7500.0, 100.0, format="%.0f")
    plt_count = st.number_input("Platelet Count (cells/mcL)", 10000.0, 1000000.0, 250000.0, 1000.0, format="%.0f")
with col3:
    neutrophile = st.number_input("Neutrophil (%)", 0.0, 100.0, 60.0, 0.1)
    eosinophile = st.number_input("Eosinophil (%)", 0.0, 100.0, 2.0, 0.1)
    monocyte = st.number_input("Monocyte (%)", 0.0, 100.0, 5.0, 0.1)

# --- ปุ่มวิเคราะห์และ Logic การเรียก API ---
if st.button("📈 วิเคราะห์ผลเลือด", use_container_width=True):
    with st.spinner("AI กำลังวิเคราะห์..."):
        patient_data = {
            "sex": sex, "age_y": age_y, "HCT": hct, "MCV": mcv, "WBC": wbc,
            "NEUTROPHILE": neutrophile, "EOSINOPHILE": eosinophile,
            "MONOCYTE": monocyte, "PLT_COUNT": plt_count
        }
        api_url = "/api/predict" # ใช้ relative path
        try:
            response = requests.post(api_url, json=patient_data, timeout=40)
            response.raise_for_status()
            result = response.json()
            predictions = result.get("predictions", {})
            
            st.subheader("📜 ผลการวินิจฉัยเบื้องต้นจากระบบ AI")
            found_conditions = [
                label.replace('is_', '').replace('_', ' ').title()
                for label, value in predictions.items() if value == 1
            ]
            if not found_conditions:
                st.success("🟢 ไม่พบความเสี่ยงที่สำคัญในเบื้องต้น")
            else:
                for condition in found_conditions:
                    st.warning(f"🔴 พบความเสี่ยง: {condition}")
            st.info("หมายเหตุ: ผลลัพธ์นี้เป็นการวิเคราะห์เบื้องต้นเท่านั้น ควรปรึกษาแพทย์เพื่อการวินิจฉัยสุดท้าย")
        except requests.exceptions.RequestException as e:
            st.error(f"เกิดข้อผิดพลาดในการเชื่อมต่อกับ AI API: {e}")