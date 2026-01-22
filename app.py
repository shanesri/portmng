import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Page Config
st.set_page_config(page_title="Shane's Macro Engine", layout="wide")

st.title("📊 Shane's Portfolio Monte Carlo Engine")
st.markdown("---")

# --- STEP 1: TICKER INPUT ---
st.subheader("Step 1: Choose Your Assets")
ticker_input = st.text_input(
    "Enter Ticker Symbols (comma separated)", 
    "BTC-USD, GLD, 2829.HK, 1617.T, EPHE, EIDO",
    help="You can find ticker symbols on Yahoo Finance (e.g., SPY, AAPL, 0005.HK)."
)

if 'tickers_ready' not in st.session_state:
    st.session_state.tickers_ready = False

if st.button("Confirm Tickers"):
    st.session_state.tickers_ready = True

# --- STEP 2 & 3: WEIGHTS & SETTINGS ---
if st.session_state.tickers_ready:
    tickers = [t.strip().upper() for t in ticker_input.split(",")]
    
    st.markdown("---")
    col_a, col_b = st.columns([1, 1])

    with col_a:
        st.subheader("Step 2: Assign Weights (%)")
        weights = []
        for ticker in tickers:
            w = st.number_input(f"Weight for {ticker} (%)", min_value=0.0, max_value=100.0, value=100.0/len(tickers), key=f"w_{ticker}")
            weights.append(w / 100.0)
        
        total_w = sum(weights)
        if not np.isclose(total_w, 1.0):
            st.warning(f"Total Weight: {total_w*100:.1f}%. Must equal 100%.")

    with col_b:
        st.subheader("Step 3: Simulation Settings")
        initial_investment = st.number_input("Initial Investment ($)", value=100, step=100)
        simulations = st.number_input("Number of Simulations", value=1000, step=100)
        time_horizon = st.number_input("Time Horizon (Days to Forecast)", value=252, step=1)
        
        # Date range for historical data
        d_col1, d_col2 = st.columns(2)
        with d_col1:
            start_date = st.date_input("Data Start Date", value=datetime.now() - timedelta(days=365*3))
        with d_col2:
            end_date = st.date_input("Data End Date", value=datetime.now())

    st.markdown("---")
    
    # --- SIMULATION EXECUTION ---
    if st.button("🚀 Run Simulation") and np.isclose(total_w, 1.0):
        try:
            with st.spinner("Processing Market Data..."):
                # Download Data
                data = yf.download(tickers, start=start_date, end=end_date)['Close']
                
                # Handling possible multi-level columns from yfinance
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                
                returns = data.pct_change().dropna()
                
                # DNA Calculations
                avg_returns = returns.mean()
                cov_matrix = returns.cov()
                
                # Show Covariance Matrix
                st.subheader("🧬 Asset Covariance (The Macro Map)")
                fig_cov, ax_cov = plt.subplots(figsize=(10, 4))
                sns.heatmap(cov_matrix, annot=True, cmap='Blues', ax=ax_cov)
                st.pyplot(fig_cov)

                # Monte Carlo Engine
                L = np.linalg.cholesky(cov_matrix)
                portfolio_sims = np.full((time_horizon, simulations), 0.0)

                for i in range(simulations):
                    Z = np.random.normal(size=(time_horizon, len(tickers)))
                    daily_returns = avg_returns.values + np.dot(Z, L.T)
                    # Compounding the growth
                    portfolio_path = np.cumprod(np.dot(daily_returns, weights) + 1) * initial_investment
                    portfolio_sims[:, i] = portfolio_path

                # Visuals & Stats
                st.markdown("---")
                st.subheader(f"📈 Portfolio Simulation: Decile Fan Chart")
                
                # --- Percentile Calculation for Fan Chart ---
                # We calculate percentiles at each time step (row)
                percentiles = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
                p_data = np.percentile(portfolio_sims, percentiles, axis=1) # Shape (len(percentiles), time_horizon)
                
                fig_fan, ax_fan = plt.subplots(figsize=(12, 6))
                time_range = np.arange(time_horizon)
                
                # Plot the fill between layers
                for i in range(len(percentiles) // 2):
                    low_idx = i
                    high_idx = len(percentiles) - 1 - i
                    # Alpha gets stronger towards the 50th percentile
                    alpha_val = 0.1 + (i * 0.1) 
                    ax_fan.fill_between(time_range, p_data[low_idx], p_data[high_idx], 
                                        color='blue', alpha=alpha_val, label=f'{percentiles[low_idx]}-{percentiles[high_idx]}%')

                # Highlight the Median (50th percentile)
                ax_fan.plot(time_range, p_data[5], color='darkblue', linewidth=2, label='Median (50th)')
                
                ax_fan.axhline(initial_investment, color='red', linestyle='--', label="Break Even")
                ax_fan.set_xlabel("Days")
                ax_fan.set_ylabel("Portfolio Value ($)")
                ax_fan.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')
                st.pyplot(fig_fan)

                # Final Metrics
                final_values = portfolio_sims[-1, :]
                median_final = np.median(final_values)
                pct_5th = np.percentile(final_values, 5)
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Median Final Value", f"${median_final:,.2f}")
                m2.metric("VaR (95% Confidence)", f"${initial_investment - pct_5th:,.2f}")
                m3.metric("Avg Final Value", f"${np.mean(final_values):,.2f}")

                # Distribution
                st.subheader("📊 Distribution of Outcomes")
                fig_hist, ax_hist = plt.subplots(figsize=(12, 4))
                ax_hist.hist(final_values, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
                ax_hist.axvline(initial_investment, color='red', linestyle='--', label='Initial')
                ax_hist.axvline(pct_5th, color='orange', linestyle='--', label='VaR')
                ax_hist.set_title("Terminal Wealth Distribution")
                ax_hist.legend()
                st.pyplot(fig_hist)

        except Exception as e:
            st.error(f"Error: {e}. Check if tickers are valid and if there's enough data for your date range.")
