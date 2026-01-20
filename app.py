import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. UI Elements
st.title("Shane's Portfolio Monte Carlo Engine")
st.sidebar.header("Portfolio Settings")

# User Inputs
ticker_input = st.sidebar.text_input("Enter Tickers (comma separated)", "SPY, GLD, TLT")
start_date = st.sidebar.date_input("Historical Start Date", value=pd.to_datetime("2021-01-01"))
initial_investment = st.sidebar.number_input("Initial Investment ($)", value=100000)

# Split and clean tickers
tickers = [t.strip().upper() for t in ticker_input.split(",")]
weights = np.array([1/len(tickers)] * len(tickers)) # Default equal weight

# 2. The Logic
try:
    # Fetch Data
    data = yf.download(tickers, start=start_date)['Adj Close']
    returns = data.pct_change().dropna()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    
    # Simulation Parameters
    simulations = 1000
    days = 252
    
    # Cholesky Decomposition
    L = np.linalg.cholesky(cov_matrix)
    portfolio_sims = np.full((days, simulations), 0.0)

    for i in range(simulations):
        Z = np.random.normal(size=(days, len(tickers)))
        daily_returns = mean_returns.values + np.dot(Z, L.T)
        portfolio_path = np.cumprod(np.dot(daily_returns, weights) + 1) * initial_investment
        portfolio_sims[:, i] = portfolio_path

    # 3. Output
    st.subheader("Simulated Future Paths")
    st.line_chart(portfolio_sims)

    # Risk Metrics
    final_values = portfolio_sims[-1, :]
    median_final = np.median(final_values)
    pct_5th = np.percentile(final_values, 5)

    st.subheader("Risk Metrics")
    st.metric("Expected Final Value (Median)", f"${median_final:,.2f}")
    st.metric("Value at Risk (95% Confidence)", f"${initial_investment - pct_5th:,.2f}")

except Exception as e:
    st.error(f"Please enter valid tickers to start the simulation.")
