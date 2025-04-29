# ============================================
# Import Libraries
# ============================================
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KernelDensity
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px

# ============================================
# Dividend Dashboard Functions
# ============================================

def display_dividend_dashboard(ticker: str):
    ticker_obj = yf.Ticker(ticker)
    info = ticker_obj.info

    st.subheader("Company Overview")
    st.write(info.get('longBusinessSummary', "No overview available."))

    st.subheader("Dividend History (Last 10 Entries)")
    dividends = ticker_obj.dividends

    if dividends.empty:
        st.write("No dividend data available for this ticker.")
    else:
        recent_dividends = dividends.tail(10)
        st.write(recent_dividends)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(recent_dividends.index, recent_dividends.values)
        ax.set_title("Dividend History (Last 10 Entries)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Dividend ($)")
        plt.xticks(rotation=45)
        st.pyplot(fig)

    st.subheader("Price History (Last 1 Year)")
    history = ticker_obj.history(period="1y")

    if history.empty:
        st.write("No price data available for this ticker.")
    else:
        st.write(history[['Close']].head())

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(history.index, history['Close'], label="Close Price")
        ax.set_title("Price History (Last 1 Year)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price ($)")
        ax.legend()
        st.pyplot(fig)

    st.subheader("Key Financial Metrics")
    eps = info.get('trailingEps')
    dividend_rate = info.get('dividendRate')
    dividend_yield = info.get('dividendYield')
    payout_ratio = (dividend_rate / eps) if eps and dividend_rate else None

    st.write("Trailing EPS:", eps if eps is not None else "N/A")
    st.write("Dividend Rate:", dividend_rate if dividend_rate is not None else "N/A")
    st.write("Dividend Yield:", dividend_yield if dividend_yield is not None else "N/A")
    st.write("Dividend Payout Ratio:", round(payout_ratio, 2) if payout_ratio else "N/A")


# ============================================
# Altman Z-Score Functions
# ============================================

def compute_altman_z(ticker: str):
    ticker_obj = yf.Ticker(ticker)
    bs = ticker_obj.balance_sheet
    fs = ticker_obj.financials
    info = ticker_obj.info

    def fetch_item(df, keys):
        for key in keys:
            for row in df.index:
                if row.strip().lower() == key.strip().lower():
                    return df.loc[row][0]
        return None

    total_assets = fetch_item(bs, ["Total Assets"])
    total_liabilities = fetch_item(bs, ["Total Liabilities", "Total Liabilities Net Minority Interest"])
    current_assets = fetch_item(bs, ["Current Assets", "Total Current Assets"])
    current_liabilities = fetch_item(bs, ["Current Liabilities", "Total Current Liabilities"])
    retained_earnings = fetch_item(bs, ["Retained Earnings"])
    ebit = fetch_item(fs, ["EBIT", "Operating Income"])
    sales = fetch_item(fs, ["Total Revenue", "Sales"])

    share_price = info.get('regularMarketPrice')
    shares_outstanding = info.get('sharesOutstanding')
    market_cap = share_price * shares_outstanding if share_price and shares_outstanding else None

    if not all([total_assets, total_liabilities, market_cap]):
        return None, "Essential data missing for computation."

    working_capital = (current_assets - current_liabilities) if current_assets and current_liabilities else 0
    ratio1 = working_capital / total_assets
    ratio2 = retained_earnings / total_assets
    ratio3 = ebit / total_assets
    ratio4 = market_cap / total_liabilities
    ratio5 = sales / total_assets

    z_score = 1.2 * ratio1 + 1.4 * ratio2 + 3.3 * ratio3 + 0.6 * ratio4 + ratio5

    if z_score > 2.99:
        classification = "Safe Zone"
    elif z_score >= 1.81:
        classification = "Grey Zone"
    else:
        classification = "Distressed Zone"

    return z_score, classification


# ============================================
# Investing Analysis Functions 
# ============================================

def extract_features(tickers):
    """
    Extracts dividend yield, price, beta, and expected return.
    """
    records = []
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            dy = info.get('dividendYield', np.nan)
            growth = info.get('earningsGrowth', np.nan)
            price = info.get('regularMarketPrice', np.nan)
            beta = info.get('beta', np.nan)
            expected_return = (dy or 0) + (growth or 0)
        except Exception:
            dy, growth, price, beta, expected_return = [np.nan] * 5
        records.append([ticker, dy, price, beta, expected_return])
    return pd.DataFrame(records, columns=['Ticker', 'Dividend Yield', 'Price', 'Stability', 'Expected Return'])


def remove_outliers(df, columns):
    """
    Removes outliers from specified columns using Z-score method.
    """
    z_scores = np.abs(stats.zscore(df[columns].dropna()))
    return df[(z_scores < 3).all(axis=1)]


def perform_clustering(df):
    """
    Clusters stocks based on selected features.
    """
    df_clean = df.dropna(subset=['Dividend Yield', 'Expected Return', 'Stability'])
    df_clean = remove_outliers(df_clean, ['Dividend Yield', 'Expected Return', 'Stability'])

    scaler = StandardScaler()
    scaled = scaler.fit_transform(df_clean[['Dividend Yield', 'Expected Return', 'Stability']])

    model = KMeans(n_clusters=3, random_state=42)
    df_clean['Cluster'] = model.fit_predict(scaled)

    return model, df_clean


def recommend_stocks(df, budget, model=None, preferences=None, min_price=20, max_price=500):
    """
    Recommends stocks within budget and preference constraints.
    """
    df_clean = df.dropna(subset=['Dividend Yield', 'Expected Return', 'Stability'])
    df_clean = remove_outliers(df_clean, ['Dividend Yield', 'Expected Return', 'Stability'])

    if preferences:
        key = preferences.get('priority')
        if key in df_clean.columns:
            df_clean = df_clean.sort_values(key, ascending=False)

    if model:
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(df_clean[['Dividend Yield', 'Expected Return', 'Stability']])
        df_clean['Cluster'] = model.predict(features_scaled)
        best_cluster = df_clean['Cluster'].mode()[0]
        df_clean = df_clean[df_clean['Cluster'] == best_cluster]

    df_clean = df_clean[(df_clean['Price'] >= min_price) & (df_clean['Price'] <= max_price)]
    selected = df_clean.head(5)
    selected['Allocation'] = budget / len(selected) if not selected.empty else 0

    return selected


def get_sp500_tickers():
    """
    Scrapes S&P 500 tickers from Wikipedia.
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table', {'id': 'constituents'})
    df = pd.read_html(str(table))[0]
    return df['Symbol'].tolist()


# ============================================
# Sector Competitor Explorer (Skeleton)
# ============================================

def sector_competitor_explorer():
    st.title("📈 Sector Competitor Explorer")

    try:
        df = pd.read_csv("your_cleaned_trimmed_df.csv")
    except Exception:
        st.error("❌ Could not load the dataset. Make sure 'your_cleaned_trimmed_df.csv' exists.")
        return

    ticker_input = st.text_input("Enter a Ticker to Find Sector Competitors", "AAPL").upper()

    if st.button("Find Competitors"):
        if ticker_input in df['ticker'].values:
            sector = df[df['ticker'] == ticker_input]['sector'].values[0]
            sector_df = df[df['sector'] == sector]
            st.write(f"Competitors in sector: {sector}")
            st.dataframe(sector_df)
        else:
            st.warning(f"Ticker {ticker_input} not found in the dataset.")
