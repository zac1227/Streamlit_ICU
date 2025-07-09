import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import shap

# ------------------------- 分頁切換 -------------------------
model_choice = st.sidebar.selectbox("請選擇要執行的模型", [
    "EOMG",
    "LOMG",
    "Thymoma",
    "Non-Thymoma"
])
#tab1, tab2, tab3, tab4 = st.tabs(["EOMG", "LOMG", "Thymoma", "Non-Thymoma"])


# ------------------------- 共用函數：預測 + SHAP -------------------------
# 共用預測＋SHAP 解釋函式
def predict_and_explain(model, x_train, input_df, model_name):
    st.subheader(" 預測結果")

    try:
        model_feature_names = model.get_booster().feature_names
        input_df = input_df[model_feature_names]
        background = x_train[model_feature_names].sample(50, random_state=42)

        # 預測
        proba = model.predict_proba(input_df)[0]
        pred_class = int(np.argmax(proba))

        if pred_class == 1:
            st.success("預測結果：ICU admission")
        else:
            st.success("預測結果：Not ICU admission")

        # SHAP 解釋（使用 kernel explainer）
        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_values = explainer.shap_values(input_df)

        st.subheader("SHAP Waterfall 解釋圖")
        fig = plt.figure()
        shap.plots.waterfall(
            shap.Explanation(
                values=shap_values[1][0],
                base_values=explainer.expected_value[1],
                data=input_df.values[0],
                feature_names=input_df.columns.tolist()
            ),
            show=False
        )
        st.pyplot(fig)

    except Exception as e:
        st.error(f"發生錯誤：{e}")



# ------------------------- 模型 A -------------------------
def run_model_a_page():
    st.title("EOMG 模型預測頁面")
    # 模型 & 資料（你之後替換正確路徑）
    import xgboost as xgb
    model = xgb.XGBClassifier()
    model.load_model(r"MG_ICU_SHAP_XGB_EOMG.json")
    x = pd.read_csv(r"MG_ICU_SHAP_Model_Data_SubGroup2_Age50D_New_FeaName.csv")
    x_train = x.drop(columns=[ "Y"])
    # 輸入變數
    Gender = st.sidebar.radio("Gender", options=[1, 2])
    BMI = st.sidebar.number_input("BMI", 10.0, 40.0, 22.5)
    Infection = st.sidebar.number_input("Infection at admission", 0, 1, 0)
    Thyroid = st.sidebar.number_input("Thyroid disease", 0, 1, 0)
    Diabetes = st.sidebar.number_input("Diabetes", 0, 1, 0)
    Hypertension = st.sidebar.number_input("Hypertension", 0, 1,0)
    ASCVD = st.sidebar.number_input("ASCVD", 0, 1, 0)
    Chronic = st.sidebar.number_input("Chronic lung disease", 0, 1, 0)
    Good = st.sidebar.number_input("Good syndrome", 0, 1, 0)
    Disease_duration= st.sidebar.number_input("Disease duration (month)", 0, 120, 0)
    MGFA = st.sidebar.number_input("MGFA clinical classification", 0, 120, 0)
    Prednisolone = st.sidebar.number_input("Prednisolone daily dose before admission", 0, 100, 0)
    Immunosuppressant = st.sidebar.number_input("Immunosuppressant at admission", 0, 1, 0)
    Anti_MuSK = st.sidebar.number_input("Anti-MuSK", 0, 1, 0)
    Anti_AChR = st.sidebar.number_input("Anti-AChR", 0, 1, 0)
    dSN = st.sidebar.number_input("dSN", 0, 1, 0)
    Thymoma = st.sidebar.number_input("Thymoma", 0, 1, 0)
    Thymic = st.sidebar.number_input("Thymic hyperplasia", 0, 1, 0)
    Thymectomy = st.sidebar.number_input("Thymectomy", 0, 1, 0)
    NLR = st.sidebar.number_input("NLR", 0.0, 20.0, 2.0)
    PLR = st.sidebar.number_input("PLR", 0.0, 1000.0, 100.0)
    LMR = st.sidebar.number_input("LMR", 0.0, 20.0, 2.0)
    SII = st.sidebar.number_input("SII", 0.0, 1000.0, 100.0)
    
    # 建立 dict（易於維護）
    input_dict = {
    "Gender": Gender,
    "BMI": BMI,
    "Infection at admission": Infection,
    "Thyroid disease": Thyroid,
    "Autoimmune disease": 0, 
    "Diabetes": Diabetes,
    "Hypertension": Hypertension,
    "ASCVD": ASCVD,
    "Chronic lung disease": Chronic,
    "Good syndrome": Good,
    "Disease duration (month)": Disease_duration,
    "MGFA clinical classification": MGFA,
    "Prednisolone daily dose before admission": Prednisolone,
    "Immunosuppressant at admission": Immunosuppressant,
    "Anti-MuSK": Anti_MuSK,
    "Anti-AChR": Anti_AChR,
    "dSN": dSN,
    "Thymoma": Thymoma,
    "Thymic hyperplasia": Thymic,
    "Thymectomy": Thymectomy,
    "NLR": NLR,
    "PLR": PLR,
    "LMR": LMR,
    "SII": SII
}

    


    if st.sidebar.button("預測模型"):
        # 用 input_dict 建立 DataFrame
       # 建立 DataFrame（按照 x_train 的欄位順序）
        input_df = pd.DataFrame([[input_dict[col] for col in x_train.columns]], columns=x_train.columns)
        # 印出模型實際特徵
        

        

        
        
        predict_and_explain(model, x_train, input_df, "模型 A")




# ------------------------- 主控制邏輯 -------------------------

if model_choice == "EOMG":
    run_model_a_page()


#with tab1:
#    run_model_a_page()
#with tab2:
#    run_model_b_page()
#with tab3:
#    run_model_c_page()
#with tab4:
#    run_model_d_page()    
