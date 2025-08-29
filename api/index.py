from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import os

# สร้าง FastAPI app
api = FastAPI(title="CBC Diagnosis API")

# --- โหลดโมเดลและเครื่องมือทั้งหมด ---
# สร้าง Path ที่ถูกต้องเพื่ออ้างอิงไฟล์จากโฟลเดอร์หลัก
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
lgbm_model = joblib.load(os.path.join(base_path, 'lgbm_model.pkl'))
dl_model = tf.keras.models.load_model(os.path.join(base_path, 'deep_learning_model.keras'))
scaler = joblib.load(os.path.join(base_path, 'scaler.pkl'))
model_columns = joblib.load(os.path.join(base_path, 'model_columns.pkl'))
    
# --- กำหนดโครงสร้างข้อมูล Input ---
class PatientData(BaseModel):
    sex: str
    age_y: int
    HCT: float
    MCV: float
    WBC: float
    NEUTROPHILE: float
    EOSINOPHILE: float
    MONOCYTE: float
    PLT_COUNT: float

# --- ฟังก์ชัน Feature Engineering (สมบูรณ์และแก้ไขแล้ว) ---
def feature_engineering_pipeline(raw_data_df, scaler_obj, model_cols):
    X_raw = raw_data_df.copy()
    numerical_features = ['HCT', 'MCV', 'WBC', 'NEUTROPHILE', 'EOSINOPHILE', 'MONOCYTE', 'PLT COUNT', 'age_y']
    for col in numerical_features:
        if X_raw[col].dtype == 'object': X_raw[col] = X_raw[col].str.replace(',', '', regex=False)
        X_raw[col] = pd.to_numeric(X_raw[col], errors='coerce')
    X_raw.fillna(X_raw.median(numeric_only=True), inplace=True)
    
    def evaluate_cbc(row):
        hct, mcv, wbc, neutrophile, eosinophile, monocyte, plt_count, sex_str = row['HCT'], row['MCV'], row['WBC'], row['NEUTROPHILE'], row['EOSINOPHILE'], row['MONOCYTE'], row['PLT COUNT'], row['sex']
        results = {}; gender_char = 'M' if sex_str == 'Male' else 'F'
        if gender_char == 'M':
            if 42 <= hct <= 54: results['HCT_status'] = "ปกติ"
            elif 33 <= hct < 42: results['HCT_status'] = "เม็ดเลือดจางเล็กน้อย"
            elif 27 <= hct < 33: results['HCT_status'] = "เม็ดเลือดจางปานกลาง"
            else: results['HCT_status'] = "เม็ดเลือดจางรุนแรง"
        elif gender_char == 'F':
            if 36 <= hct <= 48: results['HCT_status'] = "ปกติ"
            elif 33 <= hct < 36: results['HCT_status'] = "เม็ดเลือดจางเล็กน้อย"
            elif 27 <= hct < 33: results['HCT_status'] = "เม็ดเลือดจางปานกลาง"
            else: results['HCT_status'] = "เม็ดเลือดจางรุนแรง"
        if mcv < 80: results['MCV_status'] = "เม็ดเลือดแดงมีขนาดเล็ก"
        elif 80 <= mcv <= 100: results['MCV_status'] = "ปกติ"
        else: results['MCV_status'] = "เม็ดเลือดแดงมีขนาดใหญ่"
        if 6000 <= wbc <= 10000: results['WBC_status'] = "ปกติ"
        elif wbc < 6000:
            if (wbc * neutrophile / 100) < 1000: results['WBC_status'] = "เม็ดเลือดขาวต่ำอันตราย"
            else: results['WBC_status'] = "เม็ดเลือดขาวต่ำ"
        elif 10000 < wbc <= 20000: results['WBC_status'] = "เม็ดเลือดขาวสูง"
        else: results['WBC_status'] = "เม็ดเลือดขาวสูงมาก"
        if (wbc * eosinophile / 100) > 500: results['EOS_status'] = "Eosinophileสูง"
        if monocyte > 6: results['MONO_status'] = "Monocyteสูง"
        if plt_count < 100000: results['PLT_status'] = "เกล็ดเลือดต่ำ"
        elif 100000 <= plt_count <= 450000: results['PLT_status'] = "ปกติ"
        elif 450000 < plt_count <= 600000: results['PLT_status'] = "เกล็ดเลือดสูง"
        else: results['PLT_status'] = "เกล็ดเลือดสูงมาก"
        return pd.Series(results)
    
    X_rule_based = X_raw.apply(evaluate_cbc, axis=1)
    X_rule_encoded = pd.get_dummies(X_rule_based, dtype=int)
    X_raw['sex'] = X_raw['sex'].map({'Male': 0, 'Female': 1}).fillna(0)
    X_final = pd.concat([X_raw, X_rule_encoded], axis=1)
    X_processed = X_final.reindex(columns=model_cols, fill_value=0)
    features_to_scale = [col for col in numerical_features if col in X_processed.columns]
    X_processed[features_to_scale] = scaler_obj.transform(X_processed[features_to_scale])
    return X_processed

# --- สร้าง Endpoint สำหรับทำนายผล ---
@api.post("/api/predict")
def predict(patient_data: PatientData):
    patient_df = pd.DataFrame([patient_data.dict()])
    processed_features = feature_engineering_pipeline(patient_df, scaler, model_columns)
    
    # ทำนายผลด้วย Hybrid System
    pred_lgbm = lgbm_model.predict(processed_features)
    pred_dl = (dl_model.predict(processed_features) > 0.5).astype(int)
    
    labels = ['is_anemia', 'is_thalassemia_suspected', 'is_microcytic_rbc', 'is_infection_inflammation', 'is_allergy_parasite', 'is_high_lipids']
    final_prediction = {labels[i]: int(pred_lgbm[0, i]) for i in range(5)}
    final_prediction[labels[5]] = int(pred_dl[0, 5])
    
    return {"predictions": final_prediction}
