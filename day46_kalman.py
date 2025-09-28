
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Kalman Filter Demo", layout="wide")
st.title("🔹 Day 46 — Kalman Filter: Sequential Estimation for Noisy Time-Series")

st.markdown("""
Kalman Filters are recursive algorithms that estimate the state of a system from noisy measurements.  
They are widely used in **tracking, navigation, and time-series forecasting**.
""")

# --- Simulate data ---
np.random.seed(42)
n = 50
true_values = np.linspace(0, 10, n)
noise = np.random.normal(0, 0.8, n)
measurements = true_values + noise

# --- Kalman Filter Implementation ---
def kalman_filter(z, Q=1e-5, R=0.5**2):
    n = len(z)
    xhat = np.zeros(n)      # a posteriori estimate
    P = np.zeros(n)         # a posteriori error estimate
    xhatminus = np.zeros(n) # a priori estimate
    Pminus = np.zeros(n)    # a priori error estimate
    K = np.zeros(n)         # gain

    # initial guesses
    xhat[0] = 0.0
    P[0] = 1.0

    for k in range(1, n):
        # time update
        xhatminus[k] = xhat[k-1]
        Pminus[k] = P[k-1] + Q

        # measurement update
        K[k] = Pminus[k] / (Pminus[k] + R)
        xhat[k] = xhatminus[k] + K[k] * (z[k] - xhatminus[k])
        P[k] = (1 - K[k]) * Pminus[k]

    return xhat

estimated = kalman_filter(measurements)

# --- Plot Results ---
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(true_values, label="True Value", color="black", linewidth=2)
ax.scatter(range(n), measurements, label="Noisy Measurements", color="red", alpha=0.6)
ax.plot(estimated, label="Kalman Estimate", color="blue", linewidth=2)
ax.legend()
ax.set_title("Kalman Filter Estimation")
st.pyplot(fig)

st.success("✅ Kalman Filter applied successfully!")
