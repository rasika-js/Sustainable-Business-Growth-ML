import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from model import generate_dataset, train_model, predict_growth

# ----------------------------------
# Page Configuration
# ----------------------------------
st.set_page_config(
    page_title="Sustainable Growth Predictor",
    layout="wide"
)

st.title("🌱 Sustainable Business Growth Decision Support System")
st.markdown("### Machine Learning-Based Decision Support Tool")

st.markdown("---")

# ----------------------------------
# Load Dataset & Train Model
# ----------------------------------
df = generate_dataset()
model, r2, mse = train_model(df)

# ----------------------------------
# Sidebar Input Section
# ----------------------------------
st.sidebar.header("📥 Enter Sustainability Indicators")

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

    st.subheader("📊 Prediction Result")
    st.success(f"Predicted Business Growth Rate: {predicted_growth:.2f}%")

    # Visualization
    fig, ax = plt.subplots()
    ax.bar(["Predicted Growth"], [predicted_growth])
    ax.set_ylabel("Growth Rate (%)")
    ax.set_title("Predicted Sustainable Business Growth")
    st.pyplot(fig)

st.markdown("---")

# ----------------------------------
# Model Performance Section
# ----------------------------------
st.subheader("📈 Model Performance Metrics")

col1, col2 = st.columns(2)

with col1:
    st.metric("R² Score", f"{r2:.3f}")

with col2:
    st.metric("Mean Squared Error", f"{mse:.3f}")

st.markdown("---")

st.markdown("Developed using Linear Regression | Academic Demonstration Model")
