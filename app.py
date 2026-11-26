import streamlit as st
import pandas as pd
import joblib
from datetime import date

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="HomeValue | House Price Prediction",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_PATH = "house_price_model.pkl"
DATA_PATH = "data.csv"

# --------------------------------------------------
# LOAD MODEL AND DATA
# --------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_reference_data():
    try:
        return pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        return None

try:
    model = load_model()
except Exception as e:
    st.error("The trained model could not be loaded.")
    st.warning(
        "This model was saved with scikit-learn 1.6.1. "
        "Use the same version in this VS Code environment to avoid "
        "pickle compatibility errors."
    )
    st.code("python -m pip install scikit-learn==1.6.1")
    st.caption(f"Technical details: {e}")
    st.stop()

reference_df = load_reference_data()

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background-color: #F4F7FB; }
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1250px; }
[data-testid="stSidebar"] { background-color: #101D35; }
[data-testid="stSidebar"] * { color: #FFFFFF !important; }
.brand { font-size: 25px; font-weight: 800; color: #FFFFFF; padding: 12px 0 5px 0; }
.brand span { color: #38BDF8; }
.sidebar-caption { color: #AAB8CC !important; font-size: 12px; line-height: 1.7; }
.main-header {
    background: linear-gradient(120deg, #142541, #234C79);
    padding: 30px; border-radius: 18px; color: white; margin-bottom: 25px;
}
.main-header h1 { color: white; font-size: 30px; font-weight: 800; margin-bottom: 8px; }
.main-header p { color: #D5E5F6; font-size: 14px; margin-bottom: 0; }
.section-heading { color: #193451; font-size: 18px; font-weight: 700; margin: 12px 0 16px 0; }
.info-card, .result-card {
    background: white; border: 1px solid #E3EAF2; border-radius: 14px;
    padding: 18px; margin-bottom: 15px;
}
.info-label { color: #64748B; font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: .6px; }
.info-value { color: #173452; font-size: 23px; font-weight: 800; margin-top: 5px; }
.price-result {
    background: #E8F8F0; color: #087443; padding: 22px;
    border-radius: 12px; font-size: 29px; font-weight: 800; text-align: center;
}
div.stButton > button, div.stFormSubmitButton > button {
    background: linear-gradient(90deg, #087EAB, #1265A0); color: white;
    border: none; border-radius: 10px; padding: 12px; font-weight: 700;
    min-height: 48px; width: 100%;
}
div.stButton > button:hover, div.stFormSubmitButton > button:hover {
    background: #124B78; color: white; border: none;
}
div[data-testid="stForm"] {
    background: white; border: 1px solid #E2EAF2; border-radius: 16px; padding: 22px;
}
div[data-testid="stMetric"] {
    background: white; border: 1px solid #E2EAF2; border-radius: 12px; padding: 15px;
}
.footer { text-align: center; color: #718096; font-size: 12px; padding: 25px 0; }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
with st.sidebar:
    st.markdown('<div class="brand">Home<span>Value.</span></div>', unsafe_allow_html=True)
    st.markdown('<p class="sidebar-caption">HOUSE PRICE INTELLIGENCE PLATFORM</p>', unsafe_allow_html=True)
    st.divider()
    st.markdown("### Navigation")
    page = st.radio(
        "Choose a page",
        ["🏠 House Price Prediction", "📊 Model Overview"],
        label_visibility="collapsed"
    )
    st.divider()
    st.markdown("### Model Information")
    st.markdown("""
    <div class="sidebar-caption">
        Model: Selected Regression Pipeline<br><br>
        Task: Regression<br><br>
        Target: House Price<br><br>
        Dataset: House Sales<br><br>
        Version: 1.0
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    st.markdown(
        '<p class="sidebar-caption">Powered by Python, Scikit-learn & Streamlit</p>',
        unsafe_allow_html=True
    )

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1>House Price Prediction Dashboard</h1>
    <p>Estimate a property's selling price using its features and location.</p>
</div>
""", unsafe_allow_html=True)

if page == "📊 Model Overview":
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="info-card"><div class="info-label">ML Task</div><div class="info-value">Regression</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="info-card"><div class="info-label">Target Variable</div><div class="info-value">price</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="info-card"><div class="info-label">Dataset Rows</div><div class="info-value">4,600</div></div>', unsafe_allow_html=True)

    st.markdown("### Model workflow")
    st.write("Input property details → preprocessing pipeline → trained regression model → estimated house price.")
    st.markdown("### Input features")
    st.write(
        "bedrooms, bathrooms, sqft_living, sqft_lot, floors, waterfront, view, "
        "condition, sqft_above, sqft_basement, yr_built, yr_renovated, city, "
        "statezip, country, sale_year, sale_month, sale_day."
    )
    st.info("The original date column is converted into sale_year, sale_month and sale_day. The street column is not used by this model.")
    if reference_df is not None:
        st.markdown("### Dataset preview")
        st.dataframe(reference_df.head(10), use_container_width=True)
    else:
        st.caption("Add data.csv to this folder to display a dataset preview.")

else:
    st.markdown('<div class="section-heading">Enter Property Information</div>', unsafe_allow_html=True)
    st.caption("Enter the property's details below and click Predict House Price.")

    # Dataset-derived location choices, with safe fallbacks.
    if reference_df is not None:
        cities = sorted(reference_df["city"].dropna().astype(str).unique().tolist())
        zipcodes = sorted(reference_df["statezip"].dropna().astype(str).unique().tolist())
        countries = sorted(reference_df["country"].dropna().astype(str).unique().tolist())
    else:
        cities = ["Seattle"]
        zipcodes = ["WA 98101"]
        countries = ["USA"]

    with st.form("house_prediction_form"):
        st.markdown("### 🏠 Property Details")
        a, b, c = st.columns(3)
        with a:
            bedrooms = st.number_input("Bedrooms", min_value=0, max_value=20, value=3, step=1)
            bathrooms = st.number_input("Bathrooms", min_value=0.0, max_value=20.0, value=2.0, step=0.25)
            floors = st.number_input("Floors", min_value=1.0, max_value=10.0, value=1.0, step=0.5)
        with b:
            sqft_living = st.number_input("Living Area (sqft)", min_value=1, max_value=100000, value=1800, step=100)
            sqft_lot = st.number_input("Lot Area (sqft)", min_value=1, max_value=10000000, value=5000, step=100)
            sqft_above = st.number_input("Above-ground Area (sqft)", min_value=0, max_value=100000, value=1500, step=100)
        with c:
            sqft_basement = st.number_input("Basement Area (sqft)", min_value=0, max_value=100000, value=300, step=50)
            yr_built = st.number_input("Year Built", min_value=1800, max_value=2100, value=1990, step=1)
            yr_renovated = st.number_input("Year Renovated (0 = never)", min_value=0, max_value=2100, value=0, step=1)

        st.divider()
        st.markdown("### ⭐ Property Quality")
        a, b, c = st.columns(3)
        with a:
            waterfront = st.selectbox("Waterfront", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        with b:
            view = st.selectbox("View Rating", [0, 1, 2, 3, 4], index=0)
        with c:
            condition = st.selectbox("Property Condition", [1, 2, 3, 4, 5], index=2)

        st.divider()
        st.markdown("### 📍 Location")
        a, b, c = st.columns(3)
        with a:
            city = st.selectbox("City", cities)
        with b:
            statezip = st.selectbox("State / ZIP Code", zipcodes)
        with c:
            country = st.selectbox("Country", countries)

        st.divider()
        st.markdown("### 📅 Sale Date")
        sale_date = st.date_input("Expected / Recorded Sale Date", value=date(2014, 6, 1))
        submitted = st.form_submit_button("🔍 Predict House Price", use_container_width=True)

    if submitted:
        # Match the exact feature names and feature order used during training.
        input_data = pd.DataFrame([{
            "bedrooms": float(bedrooms),
            "bathrooms": float(bathrooms),
            "sqft_living": int(sqft_living),
            "sqft_lot": int(sqft_lot),
            "floors": float(floors),
            "waterfront": int(waterfront),
            "view": int(view),
            "condition": int(condition),
            "sqft_above": int(sqft_above),
            "sqft_basement": int(sqft_basement),
            "yr_built": int(yr_built),
            "yr_renovated": int(yr_renovated),
            "city": city,
            "statezip": statezip,
            "country": country,
            "sale_year": int(sale_date.year),
            "sale_month": int(sale_date.month),
            "sale_day": int(sale_date.day)
        }])

        try:
            prediction = float(model.predict(input_data)[0])
            st.markdown("---")
            st.markdown('<div class="section-heading">Prediction Result</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="price-result">Estimated House Price: ${prediction:,.2f}</div>',
                unsafe_allow_html=True
            )
            st.success("Prediction generated successfully.")
            with st.expander("View submitted property details"):
                st.dataframe(input_data, use_container_width=True)
            st.caption("This is a machine-learning estimate based on the training data, not a certified property valuation.")
        except Exception as e:
            st.error("Prediction failed. Please check that this model was trained with the same feature columns and preprocessing pipeline.")
            st.exception(e)

st.markdown("""
<div class="footer">
    HOMEVALUE | HOUSE PRICE PREDICTION PLATFORM<br>
    Machine Learning Demonstration Project
</div>
""", unsafe_allow_html=True)
