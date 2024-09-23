import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from groq import Groq
from dotenv import load_dotenv
import re
import os
from datetime import datetime, timedelta
from tabulate import tabulate

load_dotenv()


api_key = os.getenv("GROQ_API_KEY")

groq_client = Groq(api_key=api_key)

def get_top_sp500_stocks(n=20):
    tickers = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'GOOG', 'META', 'TSLA', 'BRK.B', 'JPM', 'V']
    market_caps = {ticker: yf.Ticker(ticker).info.get('marketCap', 0) for ticker in tickers}
    top_stocks = sorted(market_caps.items(), key=lambda x: x[1], reverse=True)[:n]
    return [stock[0] for stock in top_stocks]

def get_financial_data(ticker, start_date, end_date):
    stock = yf.Ticker(ticker)
    price_data = stock.history(start=start_date, end=end_date)
    income_statement = stock.financials
    return {"price_data": price_data, "income_statement": income_statement}

def format_income_statement_for_llm(income_statement_column):
    return "\n".join(f"{index}: {value:,.2f}" for index, value in income_statement_column.items())

def create_prompt_for_income_statement(current_year_income_statement, previous_year_income_statement):
    return f"""
Evaluate the following income statements for the current year and the previous year. Provide a score between 0 and 10 for each criterion. 

Income Statement for the Current Year:
{current_year_income_statement}

Income Statement for the Previous Year:
{previous_year_income_statement}

Criteria for Evaluation:
1. Revenue Growth
2. Gross Profit Margin
3. Operating Margin
4. Net Profit Margin
5. EPS Growth
6. Operating Efficiency
7. Interest Coverage Ratio
"""

def evaluate_income_statements_llm(current_year_income_statement, previous_year_income_statement):
    prompt = create_prompt_for_income_statement(current_year_income_statement, previous_year_income_statement)
    response = groq_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama3-groq-70b-8192-tool-use-preview",
        temperature=0.2,
        max_tokens=1000
    )
    analysis = response.choices[0].message.content.strip()
    score = re.search(r"Overall Score: (\d+\.\d+)", analysis)
    return float(score.group(1)) if score else None

def evaluate_stock(ticker, start_date, end_date):
    data = get_financial_data(ticker, start_date, end_date)
    income_statement = data['income_statement']
    
    scores = []
    for i in range(len(income_statement.columns) - 1):
        current_year = format_income_statement_for_llm(income_statement.iloc[:, i])
        previous_year = format_income_statement_for_llm(income_statement.iloc[:, i + 1])
        score = evaluate_income_statements_llm(current_year, previous_year)
        scores.append((income_statement.columns[i].year, score))
    
    return pd.DataFrame(scores, columns=['Year', 'Score'])

def backtest_strategy(all_scores, price_data):
    portfolio_returns = []
    executed_trades = []
    
    for year in range(min(all_scores['Year']), max(all_scores['Year']) + 1):
        top_stocks = all_scores[(all_scores['Year'] == year) & (all_scores['Score'] > 7)]
        if top_stocks.empty:
            continue
            
        top_stocks = top_stocks.nlargest(3, 'Score')['Ticker'].tolist()
        
        returns = []
        for stock in top_stocks:
            stock_prices = price_data[stock]
            start_price = stock_prices['Close'].iloc[0]
            end_price = stock_prices['Close'].iloc[-1]
            profit_loss = (end_price - start_price) / start_price
            returns.append(profit_loss)
            executed_trades.append({'Year': year, 'Ticker': stock, 'Start Price': start_price, 'End Price': end_price, 'Profit/Loss (%)': profit_loss * 100})
        
        portfolio_return = np.mean(returns) if returns else 0
        portfolio_returns.append((year, portfolio_return))
    
    cumulative_returns = pd.DataFrame(portfolio_returns, columns=['Year', 'Return'])
    cumulative_returns['Cumulative Return'] = (1 + cumulative_returns['Return']).cumprod() - 1
    
    return cumulative_returns, pd.DataFrame(executed_trades)

def plot_cumulative_returns(cumulative_returns):
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_returns['Year'], cumulative_returns['Cumulative Return'], marker='o')
    plt.title('Cumulative Returns Over Time')
    plt.xlabel('Year')
    plt.ylabel('Cumulative Return')
    st.pyplot(plt)

def calculate_sharpe_ratio(returns, risk_free_rate=0.01):
    mean_return = np.mean(returns)
    std_dev = np.std(returns)
    sharpe_ratio = (mean_return - risk_free_rate) / std_dev
    return sharpe_ratio

def moving_average(prices, window):
    return prices['Close'].rolling(window).mean()

st.title("Enhanced Stock Analysis for Algorithmic Trading")

if st.button("Get Top Stocks"):
    top_stocks = get_top_sp500_stocks(20)
    st.session_state.top_stocks = top_stocks
    st.write("Top 20 Stocks:", top_stocks)

if st.button("Evaluate Stocks"):
    if 'top_stocks' not in st.session_state or not st.session_state.top_stocks:
        st.error("Please fetch the top stocks first by clicking 'Get Top Stocks'.")
    else:
        start_date = datetime.now() - timedelta(days=5*365)
        end_date = datetime.now()
        all_scores = []
        price_data = {}
        
        for ticker in st.session_state.top_stocks:
            st.write(f"Evaluating {ticker}...")
            scores = evaluate_stock(ticker, start_date, end_date)
            scores['Ticker'] = ticker
            all_scores.append(scores)
            price_data[ticker] = get_financial_data(ticker, start_date, end_date)['price_data']

        all_scores = pd.concat(all_scores)
        all_scores['Score'] = pd.to_numeric(all_scores['Score'], errors='coerce').fillna(0)

        # Save the scores into a CSV file
        all_scores.to_csv('all_scores.csv', index=False)
        st.write(all_scores)

        # Perform backtest
        cumulative_returns, executed_trades = backtest_strategy(all_scores, price_data)

        # Print the executed trades
        st.write("Executed Trades:")
        st.table(executed_trades)

        # Print cumulative returns
        st.write("Cumulative Returns:")
        st.table(cumulative_returns)

        # Plot cumulative returns
        plot_cumulative_returns(cumulative_returns)

        # Calculate and display Sharpe Ratio
        sharpe_ratio = calculate_sharpe_ratio(cumulative_returns['Return'])
        st.write(f"Sharpe Ratio: {sharpe_ratio:.2f}")

        # Save cumulative returns and executed trades to CSV
        cumulative_returns.to_csv('backtest_results.csv', index=False)
        st.write("Backtest results saved to 'backtest_results.csv'")

        # Downloadable report
        st.download_button("Download Backtest Report", cumulative_returns.to_csv().encode('utf-8'), "backtest_results.csv", "text/csv")

        # Customizable moving average
        window = st.slider("Moving Average Window", 1, 100, 20)
        for ticker in st.session_state.top_stocks:
            if ticker in price_data:
                price_data[ticker]['MA'] = moving_average(price_data[ticker], window)
                st.line_chart(price_data[ticker][['Close', 'MA']])
