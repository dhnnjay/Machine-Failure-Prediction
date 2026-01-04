import streamlit as st
import pandas as pd
import joblib

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Predictive Maintenance System",
    page_icon="🛠️",
    layout="centered"
)

# ------------------ LOAD MODEL ------------------
model = joblib.load("gradient_boosting_model.pkl")

# ------------------ CUSTOM CSS ------------------
st.markdown("""
<style>
body {
    background-color: #f7f9fc;
}
.main-title {
    text-align: center;
    font-size: 36px;
    font-weight: 700;
}
.sub-title {
    text-align: center;
    color: #6c757d;
    margin-bottom: 30px;
}
.card {
    background-color: white;
    padding: 25px;
    border-radius: 12px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}
.footer {
    text-align: center;
    font-size: 12px;
    color: #888;
    margin-top: 30px;
}
</style>
""", unsafe_allow_html=True)

# ------------------ HEADER ------------------
st.markdown("<div class='main-title'>🛠️ Predictive Maintenance System</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='sub-title'>Machine Failure Prediction using Machine Learning</div>",
    unsafe_allow_html=True
)

# ------------------ SIDEBAR INPUTS ------------------
st.sidebar.header("⚙️ Machine Inputs")

type_option = st.sidebar.selectbox(
    "Product Type",
    ["L", "M", "H"]
)

air_temp = st.sidebar.number_input(
    "Air Temperature (K)",
    min_value=295.0,
    max_value=305.0,
    value=298.0,
    step=0.1
)

process_temp = st.sidebar.number_input(
    "Process Temperature (K)",
    min_value=305.0,
    max_value=315.0,
    value=308.0,
    step=0.1
)

rot_speed = st.sidebar.number_input(
    "Rotational Speed (rpm)",
    min_value=1000,
    max_value=3000,
    value=1500,
    step=10
)

torque = st.sidebar.number_input(
    "Torque (Nm)",
    min_value=10.0,
    max_value=80.0,
    value=40.0,
    step=0.5
)

tool_wear = st.sidebar.number_input(
    "Tool Wear (min)",
    min_value=0,
    max_value=300,
    value=50,
    step=1
)

# ------------------ ONE-HOT ENCODING ------------------
Type_H = 1 if type_option == "H" else 0
Type_L = 1 if type_option == "L" else 0
Type_M = 1 if type_option == "M" else 0

# ------------------ INPUT DATAFRAME ------------------
input_data = pd.DataFrame([{
    "Air temperature [K]": air_temp,
    "Process temperature [K]": process_temp,
    "Rotational speed [rpm]": rot_speed,
    "Torque [Nm]": torque,
    "Tool wear [min]": tool_wear,
    "Type_H": Type_H,
    "Type_L": Type_L,
    "Type_M": Type_M
}])

# ------------------ PREDICTION CARD ------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("📊 Prediction Result")

if st.button("🔍 Predict Machine Status"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    # Risk logic
    if probability < 0.30:
        risk = "LOW"
        color = "🟢"
        advice = "Machine is operating normally. Continue routine monitoring."
    elif probability < 0.60:
        risk = "MEDIUM"
        color = "🟡"
        advice = "Moderate risk detected. Schedule preventive maintenance."
    else:
        risk = "HIGH"
        color = "🔴"
        advice = "High failure risk. Immediate inspection is recommended."

    if prediction == 1:
        st.error("⚠️ Machine Failure Likely")
    else:
        st.success("✅ No Immediate Machine Failure Detected")

    st.markdown(f"""
    **Failure Probability:** `{probability:.2f}`  
    **Risk Level:** {color} **{risk}**
    """)

    st.progress(probability)
    st.info(f"**Recommended Action:** {advice}")

st.markdown("</div>", unsafe_allow_html=True)

# ------------------ FOOTER ------------------
st.markdown(
    "<div class='footer'>Gradient Boosting Model | Predictive Maintenance Project</div>",
    unsafe_allow_html=True
)
