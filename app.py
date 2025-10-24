import streamlit as st
import pandas as pd
import joblib

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Employee Attrition Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- Global Styles ----------------
st.markdown("""
<style>
/* Body Background */
.stApp {
    background: #f7fafc;  /* soft light grey */
}

/* Header Styling */
.header-container {
    background: linear-gradient(135deg, #a8edea, #fed6e3); 
    padding: 20px 20px;
    border-radius: 10px;
    text-align: center;
    color: #1f2937;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
}

.header-container h2 {
    margin-bottom: 5px;
    font-size: 26px;  /* smaller header */
}

.header-container h4 {
    margin-top: 0;
    font-size: 16px;
    font-weight: normal;
    color: #374151;
}

/* Prediction Boxes */
.prediction-box {
    padding: 20px 20px;
    border-radius: 12px;
    text-align: center;
    font-size: 18px;
    font-weight: bold;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

/* Leave - soft peach */
.leave-box {
    background: linear-gradient(135deg, #fff0f0, #ffe6e6);
    color: #b91c1c;
}

/* Stay - soft blue */
.stay-box {
    background: linear-gradient(135deg, #e0f2fe, #d0eaff);
    color: #0369a1;
}

.prediction-text {
    font-size: 14px;
    font-weight: normal;
    color: #1f2937;
}

/* Mobile adjustments */
@media (max-width: 600px) {
    .stNumberInput, .stSelectbox, .stTextInput {
        font-size: 16px !important;
        color: #1f2937 !important;
    }
}
</style>
""", unsafe_allow_html=True)

# ---------------- Header ----------------
st.markdown("""
<div class="header-container">
    <h2>🧑‍💼 Employee Attrition Predictor</h2>
    <h4>Enter employee details to predict attrition risk</h4>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ---------------- Load Model ----------------
pipeline = joblib.load("model.joblib")

# ---------------- Input Form ----------------
st.header("📝 Enter Employee Details")

with st.form(key="attrition_form"):
    col1, col2 = st.columns(2)

    # 10 features in col1
    with col1:
        age = st.number_input("Age", min_value=18, max_value=60, value=38)
        gender = st.selectbox("Gender", ["Male", "Female"])
        job_role = st.selectbox("Job Role", ["Education", "Media", "Healthcare", "Technology", "Finance"])
        years_at_company = st.number_input("Years at Company", min_value=0, max_value=51, value=15)
        monthly_income = st.number_input("Monthly Income ($)", min_value=1000, max_value=20000, value=7300, step=100)
        work_life_balance = st.selectbox("Work-Life Balance", ["Poor", "Fair", "Good", "Excellent"])
        job_satisfaction = st.selectbox("Job Satisfaction", ["Low", "Medium", "High", "Very High"])
        performance_rating = st.selectbox("Performance Rating", ["Low", "Below Average", "Average", "High"])
        number_of_promotions = st.number_input("Number of Promotions", min_value=0, max_value=10, value=1)
        overtime = st.selectbox("Overtime", ["No", "Yes"])

    # 11 features in col2
    with col2:
        distance_from_home = st.number_input("Distance from Home (miles)", min_value=0, max_value=100, value=50)
        education_level = st.selectbox("Education Level", ["High School", "Associate Degree", "Bachelor’s Degree", "Master’s Degree", "PhD"], index=2)
        marital_status = st.selectbox("Marital Status", ["Divorced", "Married", "Single"])
        number_of_dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=2)
        job_level = st.selectbox("Job Level", ["Entry", "Mid", "Senior"])
        company_size = st.selectbox("Company Size", ["Small", "Medium", "Large"])
        company_tenure = st.number_input("Company Tenure (years)", min_value=0, max_value=130, value=55)
        remote_work = st.selectbox("Remote Work", ["No", "Yes"])
        leadership_opportunities = st.selectbox("Leadership Opportunities", ["No", "Yes"])
        innovation_opportunities = st.selectbox("Innovation Opportunities", ["No", "Yes"])
        company_reputation = st.selectbox("Company Reputation", ["Poor", "Fair", "Good", "Excellent"])

    submit_button = st.form_submit_button("Predict Attrition")

# ---------------- Prediction ----------------
if submit_button:
    input_data = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "Job Role": job_role,
        "Years at Company": years_at_company,
        "Monthly Income": monthly_income,
        "Work-Life Balance": work_life_balance,
        "Job Satisfaction": job_satisfaction,
        "Performance Rating": performance_rating,
        "Number of Promotions": number_of_promotions,
        "Overtime": overtime,
        "Distance from Home": distance_from_home,
        "Education Level": education_level,
        "Marital Status": marital_status,
        "Number of Dependents": number_of_dependents,
        "Job Level": job_level,
        "Company Size": company_size,
        "Company Tenure": company_tenure,
        "Remote Work": remote_work,
        "Leadership Opportunities": leadership_opportunities,
        "Innovation Opportunities": innovation_opportunities,
        "Company Reputation": company_reputation
    }])

    prediction = pipeline.predict(input_data)[0]
    probs = pipeline.predict_proba(input_data)[0] if hasattr(pipeline, "predict_proba") else [None, None]
    prob_stay, prob_leave = probs

    # ---------------- Prediction Result Box ----------------
    if prediction == 1:
        st.markdown(f"""
        <div class="prediction-box leave-box">
            Employee Likely to Leave<br>
            <span class="prediction-text">
            Consider retention strategies and provide engagement support<br>
            Probability of Staying: {prob_stay:.2f}<br>
            Probability of Leaving: {prob_leave:.2f}
            </span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="prediction-box stay-box">
            Employee Likely to Stay<br>
            <span class="prediction-text">
            Keep supporting and motivating the employee<br>
            Probability of Staying: {prob_stay:.2f}<br>
            Probability of Leaving: {prob_leave:.2f}
            </span>
        </div>
        """, unsafe_allow_html=True)
