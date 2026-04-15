import streamlit as st
import pickle
import numpy as np
import time

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="EduPredict Pro", page_icon="🎓", layout="wide")

# --- CUSTOM CSS (For Premium Look) ---
st.markdown("""
<style>
    .main-title {
        font-size: 45px;
        font-weight: 800;
        background: -webkit-linear-gradient(#0072ff, #00c6ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: -10px;
    }
    .sub-text { font-size: 18px; color: #555; margin-bottom: 30px; }
    
    /* Custom Button Styling */
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%);
        color: black;
        width: 100%;
        border: none;
        border-radius: 12px;
        padding: 15px;
        font-size: 20px;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        transition: 0.3s;
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] { gap: 20px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; font-size: 18px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# --- LOAD MODELS ---
@st.cache_resource
def load_models():
    try:
        m = pickle.load(open('model.pkl', 'rb'))
        s = pickle.load(open('scaler.pkl', 'rb'))
        return m, s
    except Exception as e:
        return None, None

model, scaler = load_models()

if model is None or scaler is None:
    st.error("⚠️ Error: 'model.pkl' or 'scaler.pkl' not found. Please ensure both files are in the same folder.")
    st.stop()

# --- HEADER AREA ---
st.markdown('<p class="main-title">🎓 EduPredict Pro Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">AI-Powered Student Performance & Grade Analyzer</p>', unsafe_allow_html=True)
st.divider()

# --- INPUT SECTION (USING TABS) ---
tab1, tab2, tab3 = st.tabs(["👤 Personal Details", "📚 Academic Records", "🏃‍♂️ Extracurriculars"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Student Age", 15, 18, 16)
        gender = st.radio("Gender", options=[0, 1], format_func=lambda x: "Male" if x == 0 else "Female", horizontal=True)
        ethnicity = st.selectbox("Ethnicity Group", options=[0, 1, 2, 3])
    with col2:
        parental_education = st.selectbox("Parental Education Level", options=[0, 1, 2, 3, 4], help="0: None, 4: Higher Education")
        parental_support = st.selectbox("Parental Support Level", options=[0, 1, 2, 3, 4], help="0: None, 4: High Support")

with tab2:
    col3, col4 = st.columns(2)
    with col3:
        gpa = st.number_input("Current GPA (0.0 to 4.0)", min_value=0.0, max_value=4.0, value=3.0, step=0.1)
        study_time = st.slider("Weekly Study Time (Hours)", 0.0, 40.0, 10.0, step=0.5)
    with col4:
        absences = st.number_input("Total Absences (Days)", min_value=0, max_value=100, value=2)
        tutoring = st.radio("Receives Extra Tutoring?", options=[0, 1], format_func=lambda x: "No" if x==0 else "Yes", horizontal=True)

with tab3:
    st.write("Select all activities the student participates in:")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        extracurricular = st.checkbox("🎭 Extracurriculars")
    with c2:
        sports = st.checkbox("⚽ Sports")
    with c3:
        music = st.checkbox("🎸 Music")
    with c4:
        volunteering = st.checkbox("🤝 Volunteering")
    
    # Convert booleans to 1/0
    extracurricular = 1 if extracurricular else 0
    sports = 1 if sports else 0
    music = 1 if music else 0
    volunteering = 1 if volunteering else 0

st.write("") # spacing

# --- PREDICTION ACTION ---
if st.button("🚀 Analyze Performance & Predict Grade"):
    
    # Fake loading effect for premium feel
    with st.spinner("🧠 AI is analyzing student data..."):
        time.sleep(1.5)
    
    # Prepare data array (13 features)
    input_data = np.array([[age, gender, ethnicity, parental_education, study_time, 
                            absences, tutoring, parental_support, extracurricular, 
                            sports, music, volunteering, gpa]])
    
    try:
        # Scale and Predict
        scaled_features = scaler.transform(input_data)
        prediction = model.predict(scaled_features)[0]
        
        # UI Logic based on predicted grade
        st.divider()
        st.markdown("## 📊 AI Prediction Result")
        
        if prediction == 0.0:
            st.success("### 🏆 Grade: A (Excellent)")
            st.write("Outstanding academic record. Keep up the phenomenal work!")
            st.balloons()
        elif prediction == 1.0:
            st.info("### 🌟 Grade: B (Good)")
            st.write("Solid performance. With a little more effort, Grade A is within reach.")
        elif prediction == 2.0:
            st.warning("### 📚 Grade: C (Average)")
            st.write("Average standing. Focus on reducing absences and increasing study time.")
        elif prediction == 3.0:
            st.error("### ⚠️ Grade: D (Below Average)")
            st.write("High risk of failing. Immediate intervention and tutoring are recommended.")
        elif prediction == 4.0:
            st.error("### ❌ Grade: F (Failing)")
            st.write("Critical academic standing. Strong parental support and counseling needed.")
            st.snow()

        # Dashboard Metrics
        st.write("---")
        st.markdown("#### 📈 Key Performance Indicators (KPIs)")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Study Time", f"{study_time} hrs/wk")
        m2.metric("Absences", f"{absences} Days", delta="- Critical" if absences > 10 else "Normal", delta_color="inverse")
        m3.metric("Current GPA", f"{gpa}")
        m4.metric("Extracurriculars", "Active" if (extracurricular or sports or music) else "Inactive")

    except Exception as e:
        st.error(f"Prediction Error: {e}")