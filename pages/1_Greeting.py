import streamlit as st

# --- Page Config ---
st.set_page_config(page_title="Greeting & Roadmap - Shane Thailand", layout="wide")

# --- Custom CSS for Styling ---
st.markdown(
    """
    <style>
    [data-testid="stAppViewBlockContainer"] {
        max-width: 1000px !important;
        margin: 0 auto !important;
        padding-top: 2rem !important;
    }

    .intro-section {
        background-color: #1e2130;
        padding: 30px;
        border-radius: 15px;
        border: 1px solid #30363d;
        margin-bottom: 30px;
    }

    .roadmap-container {
        border-left: 2px solid #30363d;
        margin-left: 20px;
        padding-left: 30px;
        position: relative;
    }

    .phase-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 20px;
        position: relative;
    }

    .phase-card.active {
        border: 1px solid #4589ff;
        background-color: #1c2128;
    }

    .phase-badge {
        position: absolute;
        left: -41px;
        top: 20px;
        width: 20px;
        height: 20px;
        border-radius: 50%;
        background-color: #30363d;
        border: 4px solid #0e1117;
    }

    .phase-badge.active {
        background-color: #4589ff;
        box-shadow: 0 0 10px #4589ff;
    }

    .status-pill {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 10px;
        font-weight: 800;
        text-transform: uppercase;
        margin-bottom: 10px;
        background-color: #4589ff;
        color: white;
    }

    .contact-link {
        color: #4589ff;
        text-decoration: none;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Header ---
st.title("👋 Welcome to the Macro Terminal")

# --- Paragraph 1 & 2: About Me & Goal ---
st.markdown(
    """
    <div class="intro-section">
        <h3>About Me</h3>
        <p>Hi, I'm <strong>Shane Thailand</strong>. I am a <strong>CFA Level 3 Passed</strong> candidate currently looking for a high-impact financial role where I can combine my investment expertise with modern technology.</p>
        <h3>The Goal</h3>
        <p>This project is dedicated to exploring advanced financial concepts using <strong>AI-assisted coding, Python, and Streamlit</strong>. My objective is to create "Finance Made Easy" tools that simplify macro analysis, portfolio construction, and risk management.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- Paragraph 3: Roadmap ---
st.subheader("🗺️ Strategic Roadmap")
st.write("We are currently focused on laying the foundation with high-quality simulation tools.")

st.markdown(
    """
    <div class="roadmap-container">
        <!-- VER 1 -->
        <div class="phase-card active">
            <div class="phase-badge active"></div>
            <div class="status-pill">Phase 1: LIVE NOW</div>
            <h4 style="margin: 0;">VER 1: Buy & Hold MCS</h4>
            <p style="color: #8b949e; font-size: 14px; margin-top: 10px;">
                The foundation. Pick stocks, choose dates, and set weights. A robust engine simulating future 
                portfolio realities based on historical volatility and correlation DNA.
            </p>
        </div>
        <!-- VER 2 -->
        <div class="phase-card">
            <div class="phase-badge"></div>
            <h4 style="margin: 0;">VER 2: Portfolio Optimization</h4>
            <p style="color: #8b949e; font-size: 14px; margin-top: 10px;">
                Efficient Frontier integration. Automatically find weighting methods that maximize returns 
                or minimize variance, linked directly to the simulator.
            </p>
        </div>
        <!-- VER 3 -->
        <div class="phase-card">
            <div class="phase-badge"></div>
            <h4 style="margin: 0;">VER 3: Stress Testing</h4>
            <p style="color: #8b949e; font-size: 14px; margin-top: 10px;">
                Feature-rich stress tests for MCS. Test your portfolio against Black Swan events, 
                interest rate shocks, or historical crash scenarios.
            </p>
        </div>
        <!-- VER 4 -->
        <div class="phase-card">
            <div class="phase-badge"></div>
            <h4 style="margin: 0;">VER 4: Advanced Weighting Methods</h4>
            <p style="color: #8b949e; font-size: 14px; margin-top: 10px;">
                Expanding the toolkit with alternative weighting strategies like Risk Parity, 
                Inverse Volatility, and Maximum Diversification.
            </p>
        </div>
        <!-- VER 6 -->
        <div class="phase-card">
            <div class="phase-badge"></div>
            <h4 style="margin: 0;">VER 6: Non-Normal Distribution Models</h4>
            <p style="color: #8b949e; font-size: 14px; margin-top: 10px;">
                Adding support for Student-t distributions and Jump-Diffusion models to better 
                capture the "fat tails" often seen in macro markets.
            </p>
        </div>
        <!-- VER 7 -->
        <div class="phase-card">
            <div class="phase-badge"></div>
            <h4 style="margin: 0;">VER 7: Auto-Rebalancing Simulator</h4>
            <p style="color: #8b949e; font-size: 14px; margin-top: 10px;">
                Simulate the long-term impact of periodic rebalancing (Monthly/Quarterly) 
                versus buy-and-hold strategies.
            </p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- Paragraph 4: Contact ---
st.divider()
st.subheader("🤝 Let's Connect")
st.markdown(
    """
    I'm always open to discussing **job opportunities, investment strategies, or bug reports**. 
    Even if you just want to talk macro math, let's be friends!
    
    👉 **LinkedIn:** [shanesri](https://www.linkedin.com/in/shanesri/)
    """,
    unsafe_allow_html=True,
)
