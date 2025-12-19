import streamlit as st
from enum import Enum
from datetime import datetime, timedelta
import yfinance as yf
import numpy as np
from option_pricing.heatmaps import make_option_heatmaps_fig
from option_pricing import BlackScholesModel, MonteCarloPricing, BinomialTreeModel, Ticker


# -----------------------------
# Models
# -----------------------------
class OPTION_PRICING_MODEL(Enum):
    BLACK_SCHOLES = "Black Scholes Model"
    MONTE_CARLO = "Monte Carlo Simulation"
    BINOMIAL = "Cox Ross Rubenstein (CRR) Model"


# -----------------------------
# Cached data fetchers
# -----------------------------
@st.cache_data
def get_historical_data(ticker: str):
    return Ticker.get_historical_data(ticker)


@st.cache_data
def get_current_price(ticker: str):
    data = yf.Ticker(ticker).history(period="1d")
    return float(data["Close"].iloc[-1])


# -----------------------------
# Page config + styles
# -----------------------------
st.set_page_config(layout="wide", page_title="Option Pricing")

st.markdown(
    """
<style>
div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #1f2933, #111827) !important;
    color: white !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
    border-radius: 14px !important;
    padding: 0.75rem 1rem !important;
    box-shadow: 0 6px 18px rgba(0,0,0,0.35) !important;
    font-weight: 700 !important;
}

div.stButton > button[kind="primary"]:hover {
    border-color: rgba(255,255,255,0.25) !important;
    filter: brightness(1.08);
}

div.stButton > button[kind="primary"]:focus {
    outline: none !important;
    box-shadow: 0 0 0 3px rgba(147,197,253,0.35) !important;
}

.price-card {
    border-radius: 20px;
    padding: 24px 20px;
    text-align: center;
    font-family: ui-sans-serif, system-ui;
    box-shadow: 0 8px 20px rgba(0,0,0,0.25);
    color: white;
}
.price-title {
    font-size: 20px;
    margin-bottom: 10px;
    letter-spacing: 0.08em;
    opacity: 0.9;
}
.price-value {
    font-size: 46px;
    font-weight: 800;
}

section[data-testid="stSidebar"] { width: 400px !important; }

div[data-testid="stAlert"] {
    background: linear-gradient(135deg, #1f2933, #111827) !important;
    color: white !important;
    border-radius: 16px !important;
    box-shadow: 0 6px 18px rgba(0,0,0,0.35) !important;
    border: none !important;
}
div[data-testid="stAlert"] svg { fill: #93c5fd !important; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """<h1 style="text-align:left; font-size:80px;">Option pricing</h1>""",
    unsafe_allow_html=True,
)

# -----------------------------
# Session state for live updates
# -----------------------------
if "live_update" not in st.session_state:
    st.session_state.live_update = False

if "has_run" not in st.session_state:
    st.session_state.has_run = False


# -----------------------------
# Sidebar header + method picker + live toggle
# -----------------------------
with st.sidebar:
    st.markdown(
        """
<div style="
    padding: 18px 16px;
    border-radius: 16px;
    background: linear-gradient(135deg, #1f2933, #111827);
    box-shadow: 0 6px 18px rgba(0,0,0,0.35);
    margin-bottom: 16px;
">
  <div style="
      font-size: 14px;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      opacity: 0.7;
      margin-bottom: 6px;
  ">Created by</div>

  <div style="
      font-size: 22px;
      font-weight: 700;
      margin-bottom: 10px;
  ">Armon Barakchi</div>

  <div style="font-size: 16px;">
      <a href="https://github.com/ArmonBarakchi" target="_blank"
         style="text-decoration:none; margin-right:12px;">GitHub</a>
      <a href="https://www.linkedin.com/in/armon-barakchi" target="_blank"
         style="text-decoration:none;">LinkedIn</a>
  </div>
</div>
<div style="
    height: 3px;
    background: linear-gradient(to right, transparent, rgba(255,255,255,0.25), transparent);
    margin: 16px 0 8px 0;
"></div>
""",
        unsafe_allow_html=True,
    )

st.sidebar.markdown("# Option Pricing Method")
pricing_method = st.sidebar.radio(
    label="",
    options=[m.value for m in OPTION_PRICING_MODEL],
)

st.sidebar.toggle("Live update", key="live_update")

# Show chosen method
st.markdown(f"##### Pricing method: {pricing_method}")


# -----------------------------
# Sidebar inputs per model
# -----------------------------
def sidebar_common_inputs():
    """Inputs shared across models + heatmap parameters."""
    # ---------------------------
    # Core option inputs
    # ---------------------------
    ticker = st.sidebar.text_input(
        "Ticker symbol",
        "AAPL",
        help="Enter the stock symbol and press enter (e.g., AAPL for Apple Inc.)",
    )

    try:
        current_price = get_current_price(ticker)
        st.sidebar.write(f"Current price of {ticker}: ${current_price:.2f}")

        default_strike = round(current_price, 2)
        min_strike = max(0.1, round(current_price * 0.5, 2))
        max_strike = round(current_price * 2, 2)

        strike_price = st.sidebar.number_input(
            "Strike price",
            min_value=min_strike,
            max_value=max_strike,
            value=default_strike,
            step=0.01,
            help=f"The price at which the option can be exercised. Range: ${min_strike:.2f} to ${max_strike:.2f}",
        )
    except Exception:
        current_price = None
        strike_price = st.sidebar.number_input(
            "Strike price",
            min_value=0.01,
            value=100.0,
            step=0.01,
            help="Enter a valid ticker to see a suggested range.",
        )
        st.sidebar.error("Please ensure your inputted ticker is valid.")

    exercise_date = st.sidebar.date_input(
        "Exercise date",
        min_value=datetime.today() + timedelta(days=1),
        value=datetime.today() + timedelta(days=365),
        help="The date when the option can be exercised",
    )

    risk_free_rate = st.sidebar.slider(
        "Risk-free rate (%)",
        0,
        100,
        10,
        help="The theoretical rate of return of an investment with zero risk.",
    )

    sigma = st.sidebar.slider(
        "Sigma (Volatility) (%)",
        0,
        100,
        20,
        help="A measure of the stock's price variability.",
    )

    return (
        ticker,
        current_price,
        strike_price,
        exercise_date,
        risk_free_rate,
        sigma,
    )



def sidebar_mc_extras():
    number_of_simulations = st.sidebar.slider(
        "Number of simulations",
        100,
        100000,
        10000,
        help="The number of price paths to simulate. More simulations increase accuracy but take longer to compute.",
    )
    num_of_movements = st.sidebar.slider(
        "Number of price movement simulations to be visualized",
        0,
        max(1, int(number_of_simulations / 10)),
        100,
        help="The number of simulated price paths to display on the graph",
    )
    return number_of_simulations, num_of_movements


def sidebar_binomial_extras():
    number_of_time_steps = st.sidebar.slider(
        "Number of time steps",
        5000,
        100000,
        15000,
        help="The number of periods in the binomial tree. More steps increase accuracy but take longer to compute.",
    )
    return number_of_time_steps

def heatmap_params(spot_price: float):
    # ---------------------------
    # Divider
    # ---------------------------
    st.sidebar.markdown(
        """
<div style="
    height: 3px;
    background: linear-gradient(to right, transparent, rgba(255,255,255,0.25), transparent);
    margin: 18px 0 14px 0;
"></div>
""",
        unsafe_allow_html=True,
    )
    # ---------------------------
    # Heatmap section header
    # ---------------------------
    st.sidebar.markdown("# Heatmap Parameters", help = "Heatmaps visualize how option values change across spot price "
                                                       "and volatility. They provide an intuitive view of sensitivity, "
                                                       "nonlinearity, and risk under different market conditions."
                                                       "Here, the strike price and risk free rate are kept constant with the sliders above.")

    # ---------------------------
    # Heatmap inputs
    # ---------------------------
    min_spot = st.sidebar.number_input(
        "Min Spot Price",
        value=round(spot_price * 0.9, 2),
        step=0.1,
    )

    max_spot = st.sidebar.number_input(
        "Max Spot Price",
        value=round(spot_price * 1.1, 2),
        step=0.1,
    )

    min_vol = st.sidebar.slider(
        "Min Volatility for Heatmap",
        0.05,
        1.00,
        0.10,
        step=0.01,
    )

    max_vol = st.sidebar.slider(
        "Max Volatility for Heatmap",
        0.05,
        1.00,
        0.30,
        step=0.01,
    )

    return (
        min_spot,
        max_spot,
        min_vol,
        max_vol,
    )
# -----------------------------
# Calculation trigger logic
# -----------------------------
def should_run(calc_clicked: bool) -> bool:
    if calc_clicked:
        st.session_state.has_run = True
        return True
    if st.session_state.live_update and st.session_state.has_run:
        return True
    return False


# -----------------------------
# UI helpers
# -----------------------------
def show_price_cards(call_price: float, put_price: float):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f"""
<div class="price-card" style="background:#2E7D32;">
    <div class="price-title">CALL Value</div>
    <div class="price-value">${call_price:,.2f}</div>
</div>
""",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
<div class="price-card" style="background:#C62828;">
    <div class="price-title">PUT Value</div>
    <div class="price-value">${put_price:,.2f}</div>
</div>
""",
            unsafe_allow_html=True,
        )
    st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)


# -----------------------------
# Main app logic
# -----------------------------

(
    ticker,
    current_price,
    strike_price,
    exercise_date,
    risk_free_rate,
    sigma,
) = sidebar_common_inputs()

# Model-specific extras
number_of_simulations = None
num_of_movements = None
number_of_time_steps = None
if pricing_method == OPTION_PRICING_MODEL.BLACK_SCHOLES.value:
    min_spot, max_spot, min_vol, max_vol = heatmap_params(current_price)
if pricing_method == OPTION_PRICING_MODEL.MONTE_CARLO.value:
    number_of_simulations, num_of_movements = sidebar_mc_extras()

if pricing_method == OPTION_PRICING_MODEL.BINOMIAL.value:
    number_of_time_steps = sidebar_binomial_extras()


calc_clicked = st.button(
    f"Calculate option price for {ticker}",
    type="primary",
    key="calc_btn"
)



if not should_run(calc_clicked):
    st.info("Click 'Calculate option price' once. If 'Live update' is enabled, changes will update automatically.")
    st.stop()

# ---- Run pricing ----
try:
    with st.spinner("Fetching data..."):
        data = get_historical_data(ticker)

    if data is None or data.empty:
        st.error("Unable to proceed with calculations due to data fetching error.")
        st.stop()

    spot_price = Ticker.get_last_price(data, "Close")
    days_to_maturity = (exercise_date - datetime.now().date()).days
    r = risk_free_rate / 100.0
    vol = sigma / 100.0

    if pricing_method == OPTION_PRICING_MODEL.BLACK_SCHOLES.value:
        model = BlackScholesModel(spot_price, strike_price, days_to_maturity, r, vol)
        call_price = model.calculate_option_price("Call Option")
        put_price = model.calculate_option_price("Put Option")

        show_price_cards(call_price, put_price)

        fig = Ticker.plot_data(data, ticker, "Close")
        st.pyplot(fig, use_container_width=True)


        st.markdown(
            """<h1 style="text-align:left; font-size:80px;">Heatmaps</h1>""",
            unsafe_allow_html=True,
        )

        heatmap_fig = make_option_heatmaps_fig(
            K=strike_price,
            days_to_maturity=days_to_maturity,
            r=r,
            spot_min=min_spot,
            spot_max=max_spot,
            vol_min=min_vol,
            vol_max=max_vol,
            spot_points=10,  # match your example grid density
            vol_points=10,
        )
        st.pyplot(heatmap_fig, use_container_width=True)

    elif pricing_method == OPTION_PRICING_MODEL.MONTE_CARLO.value:
        model = MonteCarloPricing(spot_price, strike_price, days_to_maturity, r, vol, number_of_simulations)
        model.simulate_prices()

        call_price = model.calculate_option_price("Call Option")
        put_price = model.calculate_option_price("Put Option")

        show_price_cards(call_price, put_price)

        fig = model.plot_simulation_results(num_of_movements)
        st.pyplot(fig, use_container_width=True)

        st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)

        fig2 = Ticker.plot_data(data, ticker, "Close")
        st.pyplot(fig2, use_container_width=True)

    else:  # BINOMIAL (CRR)
        model = BinomialTreeModel(spot_price, strike_price, days_to_maturity, r, vol, number_of_time_steps)
        call_price = model.calculate_option_price("Call Option")
        put_price = model.calculate_option_price("Put Option")

        show_price_cards(call_price, put_price)

        fig = Ticker.plot_data(data, ticker, "Close")
        st.pyplot(fig, use_container_width=True)

except Exception as e:
    st.error(f"Error during calculation: {str(e)}")
