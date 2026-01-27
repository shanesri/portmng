import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# --- Page Config ---
st.set_page_config(page_title="Macro MCS Dashboard", layout="wide")

# --- Initialize Session State for Tickers ---
if 'tickers_list' not in st.session_state:
    st.session_state.tickers_list = ['BTC-USD', 'GLD', '2829.HK', '1617.T', 'EPHE', 'EIDO']

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
st.sidebar.markdown("### [📊 Monte Carlo Simulation](/)")
st.sidebar.markdown("### [🔗 About the Creator](https://shanesri.com)")

# --- Main App: Monte Carlo Simulation ---
st.title("🎲 Monte Carlo Portfolio Simulator")
st.markdown("Predicting future portfolio outcomes based on historical volatility and correlations.")

# --- Main Page Inputs ---
st.header("1. Asset Configuration")

# Ticker Input Section - Cleaner layout
col_input, col_add = st.columns([3, 1])
with col_input:
    new_ticker = st.text_input("Add Ticker", placeholder="Try GLD", key="ticker_input", label_visibility="collapsed").strip().upper()

with col_add:
    if st.button("Add Ticker", use_container_width=True):
        if new_ticker and new_ticker not in st.session_state.tickers_list:
            st.session_state.tickers_list.append(new_ticker)
            st.rerun()

# Asset Weights & Removal Section
if st.session_state.tickers_list:
    tickers = st.session_state.tickers_list
    
    # Redesigned Weight Header Row
    col_w_header, col_equal, col_spacer = st.columns([1.5, 1, 3])
    with col_w_header:
        st.subheader("Portfolio Weights")
    with col_equal:
        st.write("") # Spacer to align with subheader
        if st.button("⚖️ Equal Weight", help="Set all assets to equal weights"):
            n = len(tickers)
            if n > 0:
                eq_val = 100 // n
                for i, t in enumerate(tickers):
                    # Adjust the last one to ensure it sums exactly to 100
                    val = eq_val if i < n-1 else 100 - (eq_val * (n-1))
                    # Update session state directly to force widget refresh
                    st.session_state[f"w_val_{t}"] = str(val)
                st.rerun()

    # Grid for weights
    weight_cols = st.columns(min(len(tickers), 4))
    weights_list = []
    
    for i, ticker in enumerate(tickers):
        # Initialize session state value if not present before rendering widget
        if f"w_val_{ticker}" not in st.session_state:
            st.session_state[f"w_val_{ticker}"] = str(100 // len(tickers))

        with weight_cols[i % 4]:
            # Sub-header with ticker and remove button
            c1, c2 = st.columns([3, 1])
            with c1:
                st.markdown(f"**{ticker}**")
            with c2:
                if st.button("✖", key=f"del_{ticker}", help=f"Remove {ticker}"):
                    st.session_state.tickers_list.remove(ticker)
                    # Also clean up the weight state
                    if f"w_val_{ticker}" in st.session_state:
                        del st.session_state[f"w_val_{ticker}"]
                    st.rerun()
            
            # Text input with key tied to session state for automatic updates
            c_in, c_perc = st.columns([4, 1])
            with c_in:
                w_str = st.text_input(f"Weight for {ticker}", 
                                     key=f"w_val_{ticker}",
                                     label_visibility="collapsed")
            with c_perc:
                st.markdown("### %")
            
            try:
                weights_list.append(float(w_str))
            except ValueError:
                weights_list.append(0.0)
                
            st.write("---")

    total_weight = sum(weights_list)
    if total_weight != 100:
        st.error(f"Total weight is {total_weight}%. It must sum to 100%!")
    else:
        st.success("Weights valid (100%)")
        
        # --- Allocation Pie Chart ---
        st.write("Current Allocation Visualization")
        fig_pie, ax_pie = plt.subplots(figsize=(6, 4))
        # Filter out zero weights for the pie chart
        pie_labels = [t for t, w in zip(tickers, weights_list) if w > 0]
        pie_sizes = [w for w in weights_list if w > 0]
        
        if pie_sizes:
            ax_pie.pie(pie_sizes, labels=pie_labels, autopct='%1.1f%%', startangle=140, 
                       colors=plt.cm.Paired.colors)
            ax_pie.axis('equal') 
            st.pyplot(fig_pie)

    st.header("2. Simulation Parameters")
    s_col1, s_col2, s_col3, s_col4 = st.columns(4)

    with s_col1:
        simulations = st.slider("Number of Simulations", 100, 10000, 1000)
    with s_col2:
        time_horizon = st.number_input("Time Horizon (Days)", min_value=1, value=252)
    with s_col3:
        lookback_years = st.selectbox("Historical Range (Years)", [1, 3, 5, 10], index=1)
    with s_col4:
        initial_investment = st.number_input("Initial Investment ($)", min_value=1, value=10000)

    # --- Run Button ---
    run_sim = st.button("🚀 Run Monte Carlo Simulation", use_container_width=True)

    # --- Logic ---
    if run_sim and total_weight == 100:
        with st.spinner("Fetching data and running simulations..."):
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_years * 365)
            
            try:
                current_tickers = list(st.session_state.tickers_list)
                data = yf.download(current_tickers, start=start_date, end=end_date)['Close']
                
                missing_tickers = []
                for ticker in current_tickers:
                    if ticker not in data.columns:
                        missing_tickers.append(ticker)
                    else:
                        valid_data = data[ticker].dropna()
                        if valid_data.empty:
                            missing_tickers.append(ticker)
                        else:
                            first_date = valid_data.index[0]
                            if first_date > (start_date + timedelta(days=30)):
                                missing_tickers.append(f"{ticker} (Starts {first_date.date()})")

                if missing_tickers:
                    st.error(f"Error: Insufficient historical data for: {', '.join(missing_tickers)}. Try a shorter range or different tickers.")
                else:
                    returns = data.pct_change().dropna()
                    avg_returns = returns.mean()
                    cov_matrix = returns.cov()
                    weights = np.array(weights_list) / 100.0

                    try:
                        L = np.linalg.cholesky(cov_matrix)
                    except np.linalg.LinAlgError:
                        st.error("Correlation matrix error. Try removing highly overlapping assets.")
                        st.stop()

                    portfolio_sims = np.full((time_horizon, simulations), 0.0)
                    
                    for i in range(simulations):
                        Z = np.random.normal(size=(time_horizon, len(weights)))
                        daily_returns = avg_returns.values + np.dot(Z, L.T)
                        portfolio_path = np.cumprod(np.dot(daily_returns, weights) + 1) * initial_investment
                        portfolio_sims[:, i] = portfolio_path

                    # --- Results Visualization ---
                    st.divider()
                    st.header("3. Simulation Results")
                    
                    col_res1, col_res2 = st.columns(2)
                    final_values = portfolio_sims[-1, :]
                    mean_final = np.mean(final_values)
                    median_final = np.median(final_values)
                    var_95 = initial_investment - np.percentile(final_values, 5)

                    with col_res1:
                        st.metric("Expected Average Value", f"${mean_final:,.2f}")
                        st.metric("Value at Risk (95% Confidence)", f"${var_95:,.2f}")
                    with col_res2:
                        st.metric("Median Outcome", f"${median_final:,.2f}")
                        success_rate = (final_values > initial_investment).mean() * 100
                        st.metric("Probability of Profit", f"{success_rate:.1f}%")

                    st.subheader("Simulated Paths")
                    fig_paths, ax_paths = plt.subplots(figsize=(10, 5))
                    path_indices = np.random.choice(simulations, min(simulations, 100), replace=False)
                    ax_paths.plot(portfolio_sims[:, path_indices], color='blue', alpha=0.1)
                    ax_paths.axhline(initial_investment, color='red', linestyle='--', label='Initial')
                    ax_paths.set_xlabel("Days")
                    ax_paths.set_ylabel("Portfolio Value ($)")
                    ax_paths.grid(True, alpha=0.3)
                    st.pyplot(fig_paths)

                    c3, c4 = st.columns(2)
                    with c3:
                        st.subheader("Outcome Distribution")
                        fig_hist, ax_hist = plt.subplots()
                        ax_hist.hist(final_values, bins=50, color='skyblue', edgecolor='black')
                        ax_hist.axvline(initial_investment, color='red', linestyle='--')
                        ax_hist.set_xlabel("Final Value ($)")
                        ax_hist.set_ylabel("Frequency")
                        st.pyplot(fig_hist)
                    
                    with c4:
                        st.subheader("Asset Correlation")
                        corr = returns.corr()
                        fig_corr, ax_corr = plt.subplots()
                        im = ax_corr.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
                        fig_corr.colorbar(im)
                        ax_corr.set_xticks(np.arange(len(corr.columns)))
                        ax_corr.set_yticks(np.arange(len(corr.columns)))
                        ax_corr.set_xticklabels(corr.columns, rotation=45)
                        ax_corr.set_yticklabels(corr.columns)
                        for i in range(len(corr.columns)):
                            for j in range(len(corr.columns)):
                                ax_corr.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", color="black")
                        st.pyplot(fig_corr)

            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
else:
    st.info("Add some tickers to get started!")
