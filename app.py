import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from model import generate_dataset, train_model, predict_growth

# ----------------------------------
# Page Configuration
# ----------------------------------
st.set_page_config(
    page_title="Sustainable Growth Predictor",
    layout="wide"
)

# ----------------------------------
# Header Section
# ----------------------------------
st.markdown("""
<h2 style='text-align: center; font-weight: 600;'>
Sustainable Business Growth Decision Support System
</h2>

<p style='text-align: center; color: gray; font-size: 16px;'>
Machine Learning-Based Prediction Model
</p>

<hr style='margin-top: 10px; margin-bottom: 20px;'>
""", unsafe_allow_html=True)


# ----------------------------------
# Load Data & Train Model
# ----------------------------------
df = generate_dataset()
model, r2, mse = train_model(df)

# ----------------------------------
# Layout Columns
# ----------------------------------
col1, col2 = st.columns([1,2])

# ----------------------------------
# Sidebar Inputs
# ----------------------------------
st.sidebar.header("Sustainability Indicators")

energy_input = st.sidebar.slider("Energy Efficiency Score", 0, 100, 70)
waste_input = st.sidebar.slider("Waste Management Score", 0, 100, 70)
csr_input = st.sidebar.slider("CSR Investment Score", 0, 100, 60)
employee_input = st.sidebar.slider("Employee Satisfaction Score", 0, 100, 75)
digital_input = st.sidebar.slider("Digital Adoption Score", 0, 100, 80)

# ----------------------------------
# Prediction Section
# ----------------------------------
if st.sidebar.button("🚀 Predict Business Growth"):

    input_data = np.array([[energy_input, waste_input, csr_input, employee_input, digital_input]])
    predicted_growth = predict_growth(model, input_data)

    with col1:
        st.subheader("📊 Prediction Result")
        st.success(f"Predicted Growth Rate: {predicted_growth:.2f}%")

        # Growth Classification
        if predicted_growth < 10:
            st.warning("Sustainability Level: ⚠️ Low")
        elif predicted_growth < 15:
            st.info("Sustainability Level: 🔵 Moderate")
        else:
            st.success("Sustainability Level: 🟢 High")

    with col2:
        fig, ax = plt.subplots()
        ax.bar(["Predicted Growth"], [predicted_growth])
        ax.set_ylabel("Growth Rate (%)")
        ax.set_title("Predicted Sustainable Business Growth")
        st.pyplot(fig)

st.markdown("---")

# ----------------------------------
# Model Performance
# ----------------------------------
st.subheader("📈 Model Performance")

mcol1, mcol2 = st.columns(2)

with mcol1:
    st.metric("R² Score", f"{r2:.3f}")

with mcol2:
    st.metric("Mean Squared Error", f"{mse:.3f}")

# ----------------------------------
# Feature Importance
# ----------------------------------
st.markdown("---")
st.subheader("🔎 Feature Importance Analysis")

feature_names = df.drop("Growth", axis=1).columns
coefficients = model.coef_

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": coefficients
})

fig2, ax2 = plt.subplots()
ax2.barh(importance_df["Feature"], importance_df["Importance"])
ax2.set_xlabel("Coefficient Value")
ax2.set_title("Feature Impact on Business Growth")
st.pyplot(fig2)
