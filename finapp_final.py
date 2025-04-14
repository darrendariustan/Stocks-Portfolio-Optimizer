import streamlit as st
from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt import plotting
import copy
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go  # Add this import for candlestick charts
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from io import BytesIO # To display figures for efficient frontier later
# from statsmodels.tsa.arima.model import ARIMA
import os
# import yfinance as yf  # Replaced with yahooquery
from yahooquery import Ticker
import json
import requests
from bs4 import BeautifulSoup
import re
import matplotlib.dates as mdates
from openai import OpenAI
import pickle
import time  # Added for managing delays between API calls

# Initialize session state variables if they don't exist
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'current_ticker' not in st.session_state:
    st.session_state.current_ticker = None
if 'ticker_data' not in st.session_state:
    st.session_state.ticker_data = {}
if 'portfolio_tickers' not in st.session_state:
    st.session_state.portfolio_tickers = []
# Store portfolio optimization results
if 'portfolio_weights' not in st.session_state:
    st.session_state.portfolio_weights = {}
if 'portfolio_performance' not in st.session_state:
    st.session_state.portfolio_performance = {}
if 'portfolio_sharpe' not in st.session_state:
    st.session_state.portfolio_sharpe = None
if 'portfolio_volatility' not in st.session_state:
    st.session_state.portfolio_volatility = None
if 'portfolio_expected_return' not in st.session_state:
    st.session_state.portfolio_expected_return = None

# Function to get OpenAI API Key
def get_openai_api_key():
    api_key = st.sidebar.text_input("Enter OpenAI API Key:", type="password")
    if not api_key:
        st.sidebar.warning("Please enter your OpenAI API key to use the chatbot features.")
    return api_key

# Initialize OpenAI client
def get_openai_client(api_key):
    if api_key:
        try:
            return OpenAI(api_key=api_key)
        except TypeError:
            # Fall back to older version initialization if needed
            import openai
            openai.api_key = api_key
            return openai
    return None

# Function to extract stock ticker from user query using OpenAI's API
def extract_ticker(client, query):
    # Check for common non-ticker query patterns
    general_query_patterns = [
        'high growth', 'low risk', 'dividend', 'growth stocks', 'blue chip',
        'penny stocks', 'investment strategy', 'etf', 'index fund', 'mutual fund',
        'sector', 'industry', 'market', 'economy', 'what stocks', 'which stocks',
        'recommend', 'suggestion', 'portfolio', 'diversification'
    ]
    
    # First check if the query is likely a general question rather than about a specific ticker
    query_lower = query.lower()
    for pattern in general_query_patterns:
        if pattern in query_lower:
            log_error(f"Detected general query pattern: {pattern} in: {query}")
            return None, None
    
    # If not a general query, use OpenAI to extract ticker
    tools = [
        {
            "type": "function",
            "function": {
                "name": "extract_stock_ticker",
                "description": "Extract stock ticker symbols and company names from user queries",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ticker": {
                            "type": "string",
                            "description": "The stock ticker symbol mentioned in the query (e.g., AAPL for Apple). Return null if no specific ticker is mentioned."
                        },
                        "company_name": {
                            "type": "string",
                            "description": "The company name if mentioned (e.g., Apple). Return null if no specific company is mentioned."
                        }
                    },
                    "required": ["ticker"]
                }
            }
        }
    ]
    
    try:
        # Use OpenAI's chat completion to extract ticker
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a financial assistant that extracts stock ticker symbols from queries. Only extract actual ticker symbols. If the query is asking for recommendations, suggestions, or general categories of stocks without mentioning a specific ticker, return null."},
                {"role": "user", "content": query}
            ],
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "extract_stock_ticker"}}
        )
        
        # New format - process function calling response
        if hasattr(response, 'choices') and len(response.choices) > 0:
            message = response.choices[0].message
            if hasattr(message, 'tool_calls') and message.tool_calls:
                tool_call = message.tool_calls[0]
                function_args = json.loads(tool_call.function.arguments)
                ticker = function_args.get("ticker")
                company_name = function_args.get("company_name")
                
                # Further validation to avoid treating general terms as tickers
                if ticker and ticker.upper() in ["HIGH GROWTH", "LOW RISK", "DIVIDEND", "GROWTH", "TECH", "BLUE CHIP"]:
                    log_error(f"Prevented general term being treated as ticker: {ticker}")
                    return None, None
                    
                return ticker, company_name
        
        # If we reached here, there was an issue with extracting the ticker
        log_error(f"Failed to extract ticker from: {query}")
        return None, None
        
    except Exception as e:
        st.error(f"Error extracting ticker: {e}")
        log_error(f"Error in extract_ticker: {e}")
        return None, None

# Function to generate dummy stock data for testing
def generate_dummy_stock_data(ticker, start_date, end_date):
    """Generate simulated stock data for a ticker when real data isn't available"""
    try:
        # Convert string dates to datetime objects if they aren't already
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
            
        # Create a date range
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')
        
        # Generate random price data based on ticker (to get consistent data for the same ticker)
        np.random.seed(hash(ticker) % 2**32)
        
        # Start with a price between $10 and $100
        initial_price = np.random.uniform(10, 100)
        
        # Generate daily returns with slight upward bias
        daily_returns = np.random.normal(0.0005, 0.02, size=len(date_range))
        
        # Calculate price series
        price_series = initial_price * (1 + np.cumsum(daily_returns))
        
        # Create a DataFrame with OHLC data
        df = pd.DataFrame(index=date_range)
        df['Close'] = price_series
        df['Open'] = df['Close'].shift(1) * (1 + np.random.normal(0, 0.003, size=len(df)))
        df['High'] = np.maximum(df['Open'], df['Close']) * (1 + np.abs(np.random.normal(0, 0.005, size=len(df))))
        df['Low'] = np.minimum(df['Open'], df['Close']) * (1 - np.abs(np.random.normal(0, 0.005, size=len(df))))
        df['Volume'] = np.random.randint(100000, 5000000, size=len(df))
        
        # Fill NaN values
        df = df.fillna(method='bfill')
        
        # For consistency, convert index to date objects
        df.index = [idx.date() for idx in df.index]
        
        # Log that we used dummy data
        log_error(f"Generated dummy data for {ticker} from {start_date} to {end_date}")
        
        return df
    except Exception as e:
        log_error(f"Error generating dummy data for {ticker}: {e}")
        # Return a minimal dataframe for extreme fallback
        df = pd.DataFrame(index=[datetime.today().date()], data={'Close': [50.0]})
        return df

# Function to fetch stock data
def fetch_stock_data(ticker, period="1y"):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Create Ticker object
            stock = Ticker(ticker)
            
            # Get historical data
            history = stock.history(period=period)
            
            # Handle multi-index and reformat to match the expected structure
            if isinstance(history.index, pd.MultiIndex):
                # This selects just the data for the specific ticker
                history = history.xs(ticker, level=0, drop_level=True)
            
            # Clean index
            history = history[~history.index.duplicated(keep='first')]
            
            # Ensure index is datetime
            history.index = pd.to_datetime(history.index)
            
            # Convert datetime index to date objects for consistency
            history.index = [idx.date() for idx in history.index]
            
            # Get stock info
            info = stock.asset_profile
            # If the info is a dictionary of dictionaries (for multiple tickers), extract just this ticker's info
            if isinstance(info, dict) and ticker in info:
                info = info[ticker]
            
            # Add additional info from quote_type and summary_detail
            quote_info = stock.quote_type
            if isinstance(quote_info, dict) and ticker in quote_info:
                quote_info = quote_info[ticker]
                
            price_info = stock.summary_detail
            if isinstance(price_info, dict) and ticker in price_info:
                price_info = price_info[ticker]
            
            # Combine all info into a single dict similar to yf.Ticker.info
            combined_info = {}
            for source in [info, quote_info, price_info]:
                if isinstance(source, dict):
                    combined_info.update(source)
            
            log_error(f"Successfully fetched data for {ticker}")
            
            return {
                "history": history,
                "info": combined_info
            }
        except Exception as e:
            if attempt < max_retries - 1:
                st.warning(f"Retry {attempt+1}/{max_retries} for {ticker} after error: {e}")
                time.sleep(2)  # Wait between retries
            else:
                st.error(f"Error fetching data for {ticker} after {max_retries} attempts: {e}")
                log_error(f"Failed to get ticker '{ticker}' reason: {e}")
                return None

# Function to get stock price chart
def get_stock_price_chart(ticker, price_history):
    df = price_history.copy()
    
    # Make sure we have a Close column, or use close if available (yahooquery uses lowercase)
    if 'close' in df.columns and 'Close' not in df.columns:
        df['Close'] = df['close']
    
    fig = px.line(df, y='Close', title=f"{ticker} Stock Price")
    fig.update_layout(xaxis_title="Date", yaxis_title="Price (USD)")
    return fig

# Function to get candlestick chart
def get_candlestick_chart(ticker, price_history):
    df = price_history.copy()
    
    # Make sure we have the required columns, or use lowercase versions if available (yahooquery)
    column_map = {
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close'
    }
    
    for lower, upper in column_map.items():
        if lower in df.columns and upper not in df.columns:
            df[upper] = df[lower]
    
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name=ticker
    )])
    
    fig.update_layout(
        title=f"{ticker} Candlestick Chart",
        xaxis_title="Date",
        yaxis_title="Price (USD)"
    )
    
    return fig

# Function to get stock financial metrics
def get_financial_metrics(stock_data):
    info = stock_data["info"]
    metrics = {
        "Market Cap": info.get("marketCap", info.get("totalAssets", "N/A")),
        "P/E Ratio": info.get("trailingPE", info.get("trailingPE", "N/A")),
        "EPS": info.get("trailingEPS", info.get("epsTrailingTwelveMonths", "N/A")),
        "Dividend Yield": info.get("dividendYield", info.get("yield", "N/A")),
        "52 Week High": info.get("fiftyTwoWeekHigh", info.get("fiftyTwoWeekHigh", "N/A")),
        "52 Week Low": info.get("fiftyTwoWeekLow", info.get("fiftyTwoWeekLow", "N/A")),
        "Average Volume": info.get("averageVolume", info.get("averageDailyVolume10Day", "N/A"))
    }
    return metrics

# Function to scrape recent news about a ticker
def get_recent_news(ticker):
    try:
        url = f"https://finance.yahoo.com/quote/{ticker}/news"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        news_items = []
        for item in soup.select('h3'):
            if item.text and len(item.text.strip()) > 10:
                news_items.append(item.text.strip())
                if len(news_items) >= 5:  # Get top 5 news items
                    break
                    
        return news_items
    except Exception as e:
        return [f"Error fetching news: {e}"]

# Function to analyze stock data using LLM
def analyze_stock(client, ticker, stock_data, news):
    # Prepare context for the LLM
    info = stock_data["info"]
    history = stock_data["history"]
    
    # Calculate some additional metrics
    current_price = history['Close'].iloc[-1] if not history.empty else "N/A"
    price_change_1d = (history['Close'].iloc[-1] - history['Close'].iloc[-2]) / history['Close'].iloc[-2] * 100 if len(history) >= 2 else "N/A"
    price_change_1w = (history['Close'].iloc[-1] - history['Close'].iloc[-5]) / history['Close'].iloc[-5] * 100 if len(history) >= 5 else "N/A"
    price_change_1m = (history['Close'].iloc[-1] - history['Close'].iloc[-20]) / history['Close'].iloc[-20] * 100 if len(history) >= 20 else "N/A"
    
    # Create context
    context = f"""
    Analyzing {ticker} ({info.get('shortName', ticker)}):
    
    Current Price: ${current_price if current_price != 'N/A' else 'N/A'}
    Price Change (1 day): {price_change_1d if price_change_1d != 'N/A' else 'N/A'}%
    Price Change (1 week): {price_change_1w if price_change_1w != 'N/A' else 'N/A'}%
    Price Change (1 month): {price_change_1m if price_change_1m != 'N/A' else 'N/A'}%
    
    Key Metrics:
    - Market Cap: ${info.get('marketCap', 'N/A')}
    - P/E Ratio: {info.get('trailingPE', 'N/A')}
    - EPS: {info.get('trailingEPS', 'N/A')}
    - Dividend Yield: {info.get('dividendYield', 'N/A') if info.get('dividendYield') else 'N/A'}
    - 52 Week Range: ${info.get('fiftyTwoWeekLow', 'N/A')} - ${info.get('fiftyTwoWeekHigh', 'N/A')}
    
    Recent News Headlines:
    """
    
    for i, item in enumerate(news, 1):
        context += f"\n{i}. {item}"
    
    try:
        # Get analysis from OpenAI - new client format
        if hasattr(client, 'chat') and hasattr(client.chat, 'completions'):
            response = client.chat.completions.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system", "content": "You are a financial analyst providing insights on stocks. Give a balanced view considering recent performance, news, and metrics. Include potential risks and opportunities. Format your response with clear sections and bullet points. Keep your analysis concise, insightful, and actionable."},
                    {"role": "user", "content": context}
                ]
            )
            return response.choices[0].message.content
        # Legacy OpenAI format
        else:
            response = client.ChatCompletion.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system", "content": "You are a financial analyst providing insights on stocks. Give a balanced view considering recent performance, news, and metrics. Include potential risks and opportunities. Format your response with clear sections and bullet points. Keep your analysis concise, insightful, and actionable."},
                    {"role": "user", "content": context}
                ]
            )
            return response['choices'][0]['message']['content']
    except Exception as e:
        st.error(f"Error analyzing stock: {e}")
        return f"Sorry, I encountered an error analyzing {ticker}: {str(e)}"

# Function to calculate technical indicators
def calculate_technical_indicators(history):
    df = history.copy()
    
    # Make sure we have a Close column, or use close if available (yahooquery uses lowercase)
    if 'close' in df.columns and 'Close' not in df.columns:
        df['Close'] = df['close']
    
    # Calculate moving averages
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # Calculate Relative Strength Index (RSI)
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD (Moving Average Convergence Divergence)
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    return df

# Function to plot technical indicators
def plot_technical_indicators(df, ticker):
    # Create a figure with 2 subplots (price with MAs and RSI)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1, 1]})
    
    # Plot price and moving averages
    ax1.plot(df.index, df['Close'], label='Close Price')
    ax1.plot(df.index, df['SMA_20'], label='SMA 20')
    ax1.plot(df.index, df['SMA_50'], label='SMA 50')
    ax1.set_title(f'{ticker} Price and Technical Indicators')
    ax1.set_ylabel('Price ($)')
    ax1.legend()
    ax1.grid(True)
    
    # Plot MACD
    ax2.plot(df.index, df['MACD'], label='MACD')
    ax2.plot(df.index, df['Signal_Line'], label='Signal Line')
    ax2.bar(df.index, df['MACD'] - df['Signal_Line'], alpha=0.5, label='Histogram')
    ax2.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    ax2.set_ylabel('MACD')
    ax2.legend()
    ax2.grid(True)
    
    # Plot RSI
    ax3.plot(df.index, df['RSI'], label='RSI')
    ax3.axhline(y=70, color='r', linestyle='--', alpha=0.5)
    ax3.axhline(y=30, color='g', linestyle='--', alpha=0.5)
    ax3.set_ylabel('RSI')
    ax3.set_xlabel('Date')
    ax3.legend()
    ax3.grid(True)
    
    # Format x-axis dates
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig

# Function to process user query
def process_query(client, query):
    try:
        # Extract ticker from user query
        ticker, company_name = extract_ticker(client, query)
        
        # Check if query is about investment strategy with portfolio stocks
        investment_strategy_terms = ["investment strategy", "invest", "allocate", "allocation", "portfolio strategy", 
                                     "risk tolerance", "diversification", "diversify", "$", "dollar", "money"]
        is_investment_query = any(term in query.lower() for term in investment_strategy_terms)
        
        # Check if query is about portfolio stocks generally
        portfolio_related_terms = ["portfolio", "selected stocks", "my stocks", "these stocks", "the stocks", "i have selected"]
        is_portfolio_query = any(term in query.lower() for term in portfolio_related_terms)
        
        # If query seems to be about investment strategy and we have portfolio optimization data
        if is_investment_query and is_portfolio_query and st.session_state.portfolio_tickers:
            portfolio_tickers_str = ", ".join(st.session_state.portfolio_tickers)
            
            # Format portfolio metrics
            portfolio_metrics = ""
            if st.session_state.portfolio_sharpe is not None:
                portfolio_metrics += f"\nPortfolio Optimization Results:"
                portfolio_metrics += f"\n- Sharpe Ratio: {st.session_state.portfolio_sharpe:.2f}"
                portfolio_metrics += f"\n- Expected Annual Return: {st.session_state.portfolio_expected_return*100:.2f}%"
                portfolio_metrics += f"\n- Annual Volatility: {st.session_state.portfolio_volatility*100:.2f}%"
            
            # Format portfolio weights if available
            weights_info = ""
            if st.session_state.portfolio_weights:
                weights_info = "\nOptimized Portfolio Weights:"
                for ticker, weight in st.session_state.portfolio_weights.items():
                    weights_info += f"\n- {ticker}: {weight*100:.2f}%"
            
            prompt = f"""
            The user has selected the following stocks in their portfolio: {portfolio_tickers_str}.
            
            Their query about investment strategy is: "{query}"
            
            {portfolio_metrics}
            {weights_info}
            
            Provide a detailed investment strategy based on:
            1. The portfolio optimization results shown above (if available)
            2. The specific stocks in their portfolio
            3. Their risk tolerance mentioned in the query
            4. The investment amount mentioned in the query (if any)
            
            Include specific allocation recommendations and explain the rationale behind them.
            Consider the optimization metrics like Sharpe ratio, expected return and volatility in your recommendations.
            """
            
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000
            )
            
            analysis = response.choices[0].message.content
            return analysis
        
        # If just about portfolio generally (not investment strategy specific)
        elif is_portfolio_query and st.session_state.portfolio_tickers:
            portfolio_tickers_str = ", ".join(st.session_state.portfolio_tickers)
            prompt = f"""
            The user has selected the following stocks in their portfolio: {portfolio_tickers_str}.
            
            Their query is: "{query}"
            
            Provide a detailed analysis about these specific stocks in their portfolio.
            Include a brief overview of each stock's recent performance, potential outlook, 
            and how they might work together in a portfolio.
            """
            
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800
            )
            
            analysis = response.choices[0].message.content
            return analysis
        
        if ticker is None or ticker.upper() == 'NONE':
            # No valid ticker found, handle as a general query
            return general_query_response(client, query)
        
        # Fetch stock data using the extracted ticker symbol
        stock_data = fetch_stock_data(ticker)
        
        if stock_data is None or stock_data.get("history") is None or stock_data["history"].empty:
            # Failed to get data for the ticker
            st.error(f"Could not retrieve data for {ticker}. Please try a different query.")
            return general_query_response(client, query)
        
        # Analyze and display stock information
        analyze_stock(ticker, stock_data)
        
        # Generate a response using OpenAI based on the stock data
        prompt = f"""
        Provide a brief analysis for {ticker} ({company_name if company_name else ticker}) based on recent price action and fundamentals if available.
        If there's a specific aspect of the query: "{query}", focus on that. 
        Keep the response under 400 words and make sure it's complete and not cut off.
        """
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800
        )
        
        analysis = response.choices[0].message.content
        return analysis
        
    except Exception as e:
        st.error(f"Error processing query: {e}")
        log_error(f"Error in process_query: {e}")
        return "I encountered an error processing your request. Please try a different query."

# Function to handle general queries (when no ticker is found)
def general_query_response(client, query):
    """Handle general financial queries when no specific ticker is found."""
    
    # Check if the query is about app features
    app_feature_keywords = ["chart", "graph", "visualization", "candlestick", "line chart", 
                           "feature", "can you", "do you", "available", "function", 
                           "optimize", "portfolio", "what can", "how to use", "help"]
    
    is_app_question = any(keyword in query.lower() for keyword in app_feature_keywords)
    
    # Add context about the user's selected portfolio stocks if they exist
    portfolio_context = ""
    if st.session_state.portfolio_tickers:
        portfolio_tickers_str = ", ".join(st.session_state.portfolio_tickers)
        portfolio_context = f"The user has selected the following stocks in their portfolio: {portfolio_tickers_str}."
    
    if is_app_question:
        system_content = f"""You are a helpful assistant explaining the features of a financial app.
        This app has the following capabilities:
        
        1. Portfolio Optimization:
           - Allows users to select stocks and optimize their portfolio weights using modern portfolio theory
           - Shows efficient frontier visualization
           - Calculates expected returns, volatility, and Sharpe ratio
        
        2. Stock Visualization and Analysis:
           - Line charts showing historical stock prices
           - Candlestick charts for detailed price movement analysis
           - Technical indicators including Moving Averages, RSI, and MACD
           - Financial metrics display for each stock
        
        3. Stock News:
           - Fetches and displays recent news articles about selected stocks
        
        4. AI-powered Assistance:
           - Answers general questions about investing and finance
           - Provides analysis of specific stocks when queried
        
        {portfolio_context}
        
        When asked about features, explain the relevant functionality in a helpful way.
        Provide complete answers and make sure your responses are not cut off.
        """
    else:
        system_content = f"""You are a financial advisor specializing in stocks and investments. 
        Provide helpful, educational responses about investing, stocks, and financial markets. 
        If asked about specific stocks but can't identify a ticker, politely ask for clarification.
        
        {portfolio_context}
        
        If the user's query appears to be related to their portfolio stocks, include relevant information
        about those specific stocks in your response.
        
        Provide complete answers and make sure your responses are not cut off.
        """
    
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": query}
    ]
    
    try:
        # Use the OpenAI API to get a response
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=1200
        )
        return response.choices[0].message.content
    except Exception as e:
        log_error(f"Error in general_query_response: {e}")
        return f"I'm sorry, but I encountered an error processing your general query: {str(e)}"

# Function to plot cumulative returns
def plot_cum_returns(data, title):    
    daily_cum_returns = 1 + data.dropna().pct_change()
    daily_cum_returns = daily_cum_returns.cumprod()*100
    fig = px.line(daily_cum_returns, title=title)
    return fig
    
# Function to plot efficient frontier and max Sharpe ratio
def plot_efficient_frontier_and_max_sharpe(mu, S): 
    # Optimize portfolio for max Sharpe ratio and plot it out with efficient frontier curve
    ef = EfficientFrontier(mu, S)
    fig, ax = plt.subplots(figsize=(6,4))
    ef_max_sharpe = copy.deepcopy(ef)
    plotting.plot_efficient_frontier(ef, ax=ax, show_assets=False)

    # Find the max sharpe portfolio
    ef_max_sharpe.max_sharpe(risk_free_rate=0.02)
    ret_tangent, std_tangent, _ = ef_max_sharpe.portfolio_performance()
    ax.scatter(std_tangent, ret_tangent, marker="*", s=100, c="r", label="Max Sharpe")

    # Generate random portfolios weights using dirichlet distribution for the efficient frontier
    n_samples = 1000
    w = np.random.dirichlet(np.ones(ef.n_assets), n_samples)
    rets = w.dot(ef.expected_returns)
    stds = np.sqrt(np.diag(w @ ef.cov_matrix @ w.T)) 
    sharpes = rets / stds
    ax.scatter(stds, rets, marker=".", c=sharpes, cmap="viridis_r")

    # Output
    ax.legend()
    return fig

# Function to cache stock data
def cache_stock_data(ticker, data):
    cache_dir = "cache"
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{ticker}.pkl")
    with open(cache_file, "wb") as f:
        pickle.dump(data, f)

# Function to load cached stock data
def load_cached_stock_data(ticker):
    cache_file = os.path.join("cache", f"{ticker}.pkl")
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    return None

# Enhanced error logging function
def log_error(message):
    with open("error_log.txt", "a") as log_file:
        log_file.write(f"{datetime.now()}: {message}\n")

# Function to plot moving averages
def plot_moving_averages(ticker, price_history):
    try:
        # Create a copy of the dataframe
        df = price_history.copy()
        
        # Calculate moving averages
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA50'] = df['Close'].rolling(window=50).mean()
        df['MA200'] = df['Close'].rolling(window=200).mean()
        
        # Create the plot
        fig = go.Figure()
        
        # Add price line
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Price', line=dict(color='blue')))
        
        # Add MA lines
        fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], mode='lines', name='20-day MA', line=dict(color='orange')))
        fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], mode='lines', name='50-day MA', line=dict(color='green')))
        fig.add_trace(go.Scatter(x=df.index, y=df['MA200'], mode='lines', name='200-day MA', line=dict(color='red')))
        
        # Update layout
        fig.update_layout(title=f'{ticker} - Moving Averages', xaxis_title='Date', yaxis_title='Price (USD)',
                         legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        
        return fig
    except Exception as e:
        log_error(f"Error in plot_moving_averages for {ticker}: {e}")
        # Return an empty figure
        return go.Figure()

# Function to calculate and plot RSI
def plot_rsi(ticker, price_history, window=14):
    try:
        # Create a copy of the dataframe
        df = price_history.copy()
        
        # Calculate RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Create the plot
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                          vertical_spacing=0.1, 
                          row_heights=[0.7, 0.3])
        
        # Add price to the first subplot
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Price', line=dict(color='blue')), row=1, col=1)
        
        # Add RSI to the second subplot
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode='lines', name='RSI', line=dict(color='purple')), row=2, col=1)
        
        # Add overbought/oversold lines
        fig.add_trace(go.Scatter(x=df.index, y=[70]*len(df), mode='lines', name='Overbought', 
                               line=dict(color='red', dash='dash')), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=[30]*len(df), mode='lines', name='Oversold', 
                               line=dict(color='green', dash='dash')), row=2, col=1)
        
        # Update layout
        fig.update_layout(title=f'{ticker} - RSI (14)', xaxis_title='Date', height=600,
                         legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        fig.update_yaxes(title_text='Price (USD)', row=1, col=1)
        fig.update_yaxes(title_text='RSI', row=2, col=1)
        
        return fig
    except Exception as e:
        log_error(f"Error in plot_rsi for {ticker}: {e}")
        # Return an empty figure
        return go.Figure()

# Function to calculate and plot MACD
def plot_macd(ticker, price_history):
    try:
        # Create a copy of the dataframe
        df = price_history.copy()
        
        # Calculate MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        histogram = macd - signal
        
        df['MACD'] = macd
        df['Signal'] = signal
        df['Histogram'] = histogram
        
        # Create the plot
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                          vertical_spacing=0.1, 
                          row_heights=[0.7, 0.3])
        
        # Add price to the first subplot
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Price', line=dict(color='blue')), row=1, col=1)
        
        # Add MACD to the second subplot
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], mode='lines', name='MACD', line=dict(color='blue')), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Signal'], mode='lines', name='Signal', line=dict(color='orange')), row=2, col=1)
        
        # Add histogram as bar chart
        colors = ['red' if val < 0 else 'green' for val in df['Histogram']]
        fig.add_trace(go.Bar(x=df.index, y=df['Histogram'], name='Histogram', marker_color=colors), row=2, col=1)
        
        # Update layout
        fig.update_layout(title=f'{ticker} - MACD', xaxis_title='Date', height=600,
                         legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        fig.update_yaxes(title_text='Price (USD)', row=1, col=1)
        fig.update_yaxes(title_text='MACD', row=2, col=1)
        
        return fig
    except Exception as e:
        log_error(f"Error in plot_macd for {ticker}: {e}")
        # Return an empty figure
        return go.Figure()

# Function to normalize yahooquery data
def normalize_yahoo_data(df):
    """
    Normalize yahooquery data to ensure consistent column names and index format
    """
    # Make a copy to avoid modifying the original dataframe
    df = df.copy()
    
    # Handle column name differences (yahooquery uses lowercase)
    column_map = {
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume',
        'adjclose': 'Adj Close'
    }
    
    # Normalize column names
    for lower, upper in column_map.items():
        if lower in df.columns and upper not in df.columns:
            df.rename(columns={lower: upper}, inplace=True)
    
    # Ensure index is datetime
    if len(df) > 0:
        df.index = pd.to_datetime(df.index)
        # Convert datetime index to date objects for consistency
        df.index = [idx.date() for idx in df.index]
    
    return df

# Update the fetch_stock_data function to use the normalizer
def fetch_stock_data(ticker, period="1y"):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Create Ticker object
            stock = Ticker(ticker)
            
            # Get historical data
            history = stock.history(period=period)
            
            # Handle multi-index and reformat to match the expected structure
            if isinstance(history.index, pd.MultiIndex):
                # This selects just the data for the specific ticker
                history = history.xs(ticker, level=0, drop_level=True)
            
            # Clean index
            history = history[~history.index.duplicated(keep='first')]
            
            # Normalize data to ensure consistent column names and format
            history = normalize_yahoo_data(history)
            
            # Ensure index is datetime
            history.index = pd.to_datetime(history.index)
            
            # Convert datetime index to date objects for consistency
            history.index = [idx.date() for idx in history.index]
            
            # Get stock info
            info = stock.asset_profile
            # If the info is a dictionary of dictionaries (for multiple tickers), extract just this ticker's info
            if isinstance(info, dict) and ticker in info:
                info = info[ticker]
            
            # Add additional info from quote_type and summary_detail
            quote_info = stock.quote_type
            if isinstance(quote_info, dict) and ticker in quote_info:
                quote_info = quote_info[ticker]
                
            price_info = stock.summary_detail
            if isinstance(price_info, dict) and ticker in price_info:
                price_info = price_info[ticker]
            
            # Combine all info into a single dict similar to yf.Ticker.info
            combined_info = {}
            for source in [info, quote_info, price_info]:
                if isinstance(source, dict):
                    combined_info.update(source)
            
            log_error(f"Successfully fetched data for {ticker}")
            
            return {
                "history": history,
                "info": combined_info
            }
        except Exception as e:
            if attempt < max_retries - 1:
                st.warning(f"Retry {attempt+1}/{max_retries} for {ticker} after error: {e}")
                time.sleep(2)  # Wait between retries
            else:
                st.error(f"Error fetching data for {ticker} after {max_retries} attempts: {e}")
                log_error(f"Failed to get ticker '{ticker}' reason: {e}")
                return None

# Function to analyze stock data
def analyze_stock(ticker, stock_data):
    if not stock_data:
        st.error(f"No data available for {ticker}")
        return
    
    try:
        price_history = stock_data["history"]
        info = stock_data["info"]
        
        # Check if we have the necessary data
        if price_history.empty:
            st.error(f"No price history available for {ticker}")
            return
        
        # Get company name from info, with ticker as fallback
        company_name = info.get("longName", info.get("shortName", ticker))
        
        # Display the company name and ticker symbol
        st.subheader(f"{company_name} ({ticker})")
        
        # Show company business summary if available
        if "longBusinessSummary" in info:
            with st.expander("Company Description"):
                st.write(info["longBusinessSummary"])
        elif "description" in info:
            with st.expander("Company Description"):
                st.write(info["description"])
        
        # Display stock price chart
        st.write("## Stock Price Chart")
        chart_tabs = st.tabs(["Line Chart", "Candlestick Chart"])
        
        with chart_tabs[0]:
            fig = get_stock_price_chart(ticker, price_history)
            st.plotly_chart(fig)
        
        with chart_tabs[1]:
            fig = get_candlestick_chart(ticker, price_history)
            st.plotly_chart(fig)
        
        # Calculate and display key financial metrics
        st.write("## Key Financial Metrics")
        metrics = get_financial_metrics(stock_data)
        
        # Format metrics into three columns
        col1, col2, col3 = st.columns(3)
        metrics_list = list(metrics.items())
        
        # Split metrics across columns
        for i, (key, value) in enumerate(metrics_list):
            with [col1, col2, col3][i % 3]:
                if key == "Market Cap" and isinstance(value, (int, float)):
                    value = f"${value/1e9:.2f}B" if value >= 1e9 else f"${value/1e6:.2f}M"
                elif key == "Dividend Yield" and isinstance(value, (int, float)):
                    value = f"{value*100:.2f}%" if value else 'N/A'
                st.metric(key, value)
        
        # Recent News
        st.write("## Recent News")
        news = get_recent_news(ticker)
        if news:
            for article in news[:5]:  # Display up to 5 news articles
                st.markdown(f"### [{article}]({article})")
                # st.write(f"Source: {article['source']} | {article['date']}")
                # st.write(article['summary'])
                st.write("---")
        else:
            st.write("No recent news available")
        
        # Technical Analysis
        st.write("## Technical Analysis")
        tech_tabs = st.tabs(["Moving Averages", "RSI", "MACD"])
        
        with tech_tabs[0]:
            fig = plot_moving_averages(ticker, price_history)
            st.plotly_chart(fig)
        
        with tech_tabs[1]:
            fig = plot_rsi(ticker, price_history)
            st.plotly_chart(fig)
        
        with tech_tabs[2]:
            fig = plot_macd(ticker, price_history)
            st.plotly_chart(fig)
            
    except Exception as e:
        st.error(f"Error analyzing {ticker}: {e}")
        log_error(f"Error analyzing {ticker}: {e}")

st.set_page_config(page_title = "Darren's Stock Portfolio Optimizer & Advisor", layout = "wide")
st.header("Darren's Stock Portfolio Optimizer & AI Stock Advisor")

# Set up the tabs for different features
tabs = st.tabs(["Portfolio Optimizer", "AI Stock Advisor"])

# Portfolio Optimizer tab
with tabs[0]:
    col1, col2 = st.columns(2)
    
    with col1:
        start_date = st.date_input("Start Date",datetime(2013, 1, 1))
        
    with col2:
        end_date = st.date_input("End Date", datetime(2024, 1, 1)) # it defaults to current date
    
    # Define the list of available tickers
    available_tickers = ["BRK-A", "DNUT", "DPZ", "LKNCY", "MCD", "PZZA", "QSR", "SBUX", "WEN", "YUM"]
    
    # Use a multiselect widget to select tickers
    selected_tickers = st.multiselect('Select tickers for your portfolio', options=available_tickers, default=[])
    
    # Store selected tickers in session state
    st.session_state.portfolio_tickers = selected_tickers
    
    # Function to load stock data for the portfolio optimizer
    def load_portfolio_data(tickers, start_date, end_date):
        stocks_df = pd.DataFrame()
        successful_tickers = []
        use_dummy_data = st.checkbox("Use simulated data if real data unavailable", value=True, 
                                    help="When checked, the app will use simulated data if Yahoo Finance API fails")
        
        if not tickers:
            st.warning("Please select at least 2 tickers to build a portfolio.")
            return None, []
        
        # Fetch stock prices for each selected ticker
        for ticker in tickers:
            try:
                # Try loading cached data first
                cached_data = load_cached_stock_data(ticker)
                if cached_data is not None:
                    # Display first and last available dates for cached data
                    first_date = cached_data.index.min()
                    last_date = cached_data.index.max()
                    st.write(f"Loaded cached data for {ticker}")
                    st.write(f"{ticker}: First available date: {first_date}, Last available date: {last_date}")
                    
                    stocks_df[ticker] = cached_data['Close']
                    successful_tickers.append(ticker)
                    continue  # Skip to next ticker

                # Add a delay between requests to avoid rate limiting
                time.sleep(2)  # Delay to reduce rate limiting
                
                # Try multiple times with error handling
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        # Create a Ticker object for this symbol
                        stock = Ticker(ticker)
                        
                        # Fetch stock data with proper date parameters
                        stock_data = stock.history(
                            start=start_date,
                            end=end_date,
                        )
                        
                        # Handle multi-index result (yahooquery returns MultiIndex with ticker as first level)
                        if isinstance(stock_data.index, pd.MultiIndex):
                            # This selects just the data for the specific ticker
                            stock_data = stock_data.xs(ticker, level=0, drop_level=True)
                        
                        if stock_data.empty:
                            if attempt < max_retries - 1:
                                time.sleep(3)  # Increased delay between retries
                                continue
                            elif use_dummy_data:
                                # Generate dummy data if API fails after all retries
                                stock_data = generate_dummy_stock_data(ticker, start_date, end_date)
                                log_error(f"Using dummy data for {ticker} after failed API calls")
                            else:
                                error_message = f"Could not retrieve data for {ticker} after {max_retries} attempts"
                                st.error(error_message)
                                log_error(error_message)
                                break
                        
                        # Clean and process the data - fix for index issues
                        stock_data = stock_data[~stock_data.index.duplicated(keep='first')]
                        
                        # Ensure index is datetime
                        stock_data.index = pd.to_datetime(stock_data.index)
                        
                        # Convert datetime index to date objects for consistency
                        stock_data.index = [idx.date() for idx in stock_data.index]
                        
                        # Check for Close column and add to dataframe
                        if 'close' in stock_data.columns:
                            # Normalize column names (yahooquery uses lowercase)
                            stock_data.rename(columns={'close': 'Close'}, inplace=True)
                        
                        if 'Close' in stock_data.columns:
                            first_date = stock_data.index.min()
                            last_date = stock_data.index.max()
                            st.write(f"{ticker}: First available date: {first_date}, Last available date: {last_date}")
                
                            stocks_df[ticker] = stock_data['Close']
                            successful_tickers.append(ticker)
                            
                            # Cache the data to reduce future API calls
                            cache_stock_data(ticker, stock_data)
                            break  # Success, exit retry loop
                        elif use_dummy_data:
                            # Generate dummy data if Close column is missing
                            stock_data = generate_dummy_stock_data(ticker, start_date, end_date)
                            stocks_df[ticker] = stock_data['Close']
                            successful_tickers.append(ticker)
                            log_error(f"Using dummy data for {ticker} - missing Close column")
                            break
                        else:
                            if attempt < max_retries - 1:
                                time.sleep(3)
                                continue
                            else:
                                error_message = f"No 'Close' column found for {ticker} after {max_retries} attempts"
                                st.error(error_message)
                                log_error(error_message)
                    except Exception as e:
                        if attempt < max_retries - 1:
                            st.warning(f"Retry {attempt+1}/{max_retries} for {ticker}: {e}")
                            time.sleep(3)  # Increased delay
                        elif use_dummy_data:
                            # Generate dummy data if exception occurs
                            stock_data = generate_dummy_stock_data(ticker, start_date, end_date)
                            stocks_df[ticker] = stock_data['Close']
                            successful_tickers.append(ticker)
                            error_message = f"Using dummy data for {ticker} after error: {e}"
                            st.warning(error_message)
                            log_error(error_message)
                            break
                        else:
                            error_message = f"Error loading data for {ticker}: {e}"
                            st.error(error_message)
                            log_error(error_message)
            except Exception as e:
                if use_dummy_data:
                    # Generate dummy data for unexpected errors
                    stock_data = generate_dummy_stock_data(ticker, start_date, end_date)
                    stocks_df[ticker] = stock_data['Close']
                    successful_tickers.append(ticker)
                    error_message = f"Using dummy data for {ticker} after unexpected error: {e}"
                    st.warning(error_message)
                    log_error(error_message)
                else:
                    error_message = f"Unexpected error processing {ticker}: {e}"
                    st.error(error_message)
                    log_error(error_message)
        
        # Check if we have enough data
        if len(successful_tickers) < 2:
            st.error("Not enough tickers with valid data. Please select more tickers.")
            return None, successful_tickers
        
        # Filter by date range
        if not stocks_df.empty:
            filtered_df = stocks_df[(stocks_df.index >= start_date) & (stocks_df.index <= end_date)]
            
            if filtered_df.empty:
                st.error("No data available for the selected date range.")
                return None, successful_tickers
            
            return filtered_df, successful_tickers
        
        return None, successful_tickers
    
    # Load the portfolio data
    stocks_df, successful_tickers = load_portfolio_data(selected_tickers, start_date, end_date)
    
    # Only continue with optimization if we have valid data
    if stocks_df is not None and not stocks_df.empty and len(successful_tickers) >= 2:
        # Display the stocks_df DataFrame
        st.write("Data Loaded:")
        st.write(stocks_df)
        
        # Plot Individual Stock Prices
        fig_price = px.line(stocks_df, title='Price of Individual Stocks')
    
        # Plot Individual Cumulative Returns
        fig_cum_returns = plot_cum_returns(stocks_df, 'Cumulative Returns of Individual Stocks Starting with $100')
    
        # Calculate and Plot Correlation Matrix between Stocks
        corr_df = stocks_df.corr().round(2)
        fig_corr = px.imshow(corr_df, text_auto=True, title = 'Correlation between Stocks')
            
        # Calculate expected returns and sample covariance matrix for portfolio optimization later
        mu = expected_returns.mean_historical_return(stocks_df)
        S = risk_models.sample_cov(stocks_df)
        
        # Plot efficient frontier curve
        fig = plot_efficient_frontier_and_max_sharpe(mu, S)
        fig_efficient_frontier = BytesIO()
        fig.savefig(fig_efficient_frontier, format="png")
        
        # Get optimized weights
        ef = EfficientFrontier(mu, S)
        risk_free_rate = 0.02  # Define risk-free rate as a variable
        ef.max_sharpe(risk_free_rate=risk_free_rate)  # Use the variable
        weights = ef.clean_weights()
        expected_annual_return, annual_volatility, sharpe_ratio = ef.portfolio_performance(risk_free_rate=risk_free_rate)  # Use the same variable
        weights_df = pd.DataFrame.from_dict(weights, orient = 'index')
        weights_df.columns = ['weights']
        
        # Store portfolio optimization results in session state
        st.session_state.portfolio_weights = weights
        st.session_state.portfolio_performance = {
            "Expected Annual Return": expected_annual_return,
            "Annual Volatility": annual_volatility,
            "Sharpe Ratio": sharpe_ratio
        }
        st.session_state.portfolio_sharpe = sharpe_ratio
        st.session_state.portfolio_volatility = annual_volatility
        st.session_state.portfolio_expected_return = expected_annual_return
        
        # Calculate returns of portfolio with optimized weights
        stocks_df['Optimized Portfolio'] = 0
        for ticker, weight in weights.items():
            stocks_df['Optimized Portfolio'] += stocks_df[ticker]*weight
        
        # Plot Cumulative Returns of Optimized Portfolio
        fig_cum_returns_optimized = plot_cum_returns(stocks_df['Optimized Portfolio'], 'Cumulative Returns of Optimized Portfolio Starting with $100')
        
        # Display everything on Streamlit
        st.subheader("Your Portfolio Consists of {} Stocks".format(", ".join(successful_tickers)))
        st.plotly_chart(fig_cum_returns_optimized)
        
        st.subheader("Optimized Max Sharpe Portfolio Weights")
        st.dataframe(weights_df)
        
        st.subheader("Optimized Max Sharpe Portfolio Performance")
        st.image(fig_efficient_frontier)
        
        # Visualization when starting with $100
        st.subheader('Expected annual return: {}%'.format((expected_annual_return*100).round(2)))
        st.subheader('Annual volatility: {}%'.format((annual_volatility*100).round(2)))
        st.subheader('Sharpe Ratio: {}'.format(sharpe_ratio.round(2)))
        
        st.plotly_chart(fig_corr)
        st.plotly_chart(fig_price)
        st.plotly_chart(fig_cum_returns)
        
        # Add Technical Indicators section
        st.subheader("Technical Indicators for Portfolio Stocks")
        
        # Create tabs for each technical indicator type
        tech_tabs = st.tabs(["Moving Averages", "RSI", "MACD"])
        
        # For each stock, get historical data and plot technical indicators
        for ticker in successful_tickers:
            # Get stock data for this ticker
            stock_data = fetch_stock_data(ticker, period="1y")
            
            if stock_data and "history" in stock_data and not stock_data["history"].empty:
                price_history = stock_data["history"]
                
                # Moving Averages tab
                with tech_tabs[0]:
                    try:
                        st.subheader(f"{ticker} - Moving Averages")
                        # Create a copy of the dataframe
                        df = price_history.copy()
                        
                        # Calculate moving averages
                        df['SMA20'] = df['Close'].rolling(window=20).mean()
                        df['SMA50'] = df['Close'].rolling(window=50).mean()
                        df['SMA200'] = df['Close'].rolling(window=200).mean()
                        
                        # Create the plot
                        fig = px.line(title=f"{ticker} - Moving Averages")
                        fig.add_scatter(x=df.index, y=df['Close'], name='Close Price')
                        fig.add_scatter(x=df.index, y=df['SMA20'], name='20-Day SMA')
                        fig.add_scatter(x=df.index, y=df['SMA50'], name='50-Day SMA')
                        fig.add_scatter(x=df.index, y=df['SMA200'], name='200-Day SMA')
                        
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error plotting moving averages for {ticker}: {e}")
                
                # RSI tab
                with tech_tabs[1]:
                    try:
                        st.subheader(f"{ticker} - Relative Strength Index")
                        # Create a copy of the dataframe
                        df = price_history.copy()
                        
                        # Calculate RSI
                        delta = df['Close'].diff()
                        gain = delta.where(delta > 0, 0)
                        loss = -delta.where(delta < 0, 0)
                        
                        avg_gain = gain.rolling(window=14).mean()
                        avg_loss = loss.rolling(window=14).mean()
                        rs = avg_gain / avg_loss
                        df['RSI'] = 100 - (100 / (1 + rs))
                        
                        # Create the plot
                        fig = px.line(title=f"{ticker} - Relative Strength Index (RSI)")
                        fig.add_scatter(x=df.index, y=df['RSI'], name='RSI')
                        
                        # Add reference lines for overbought (70) and oversold (30) levels
                        fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
                        fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
                        
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error plotting RSI for {ticker}: {e}")
                
                # MACD tab
                with tech_tabs[2]:
                    try:
                        st.subheader(f"{ticker} - MACD")
                        # Create a copy of the dataframe
                        df = price_history.copy()
                        
                        # Calculate the 12-day EMA
                        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
                        
                        # Calculate the 26-day EMA
                        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
                        
                        # Calculate the MACD line
                        df['MACD'] = ema12 - ema26
                        
                        # Calculate the signal line (9-day EMA of MACD)
                        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
                        
                        # Calculate the histogram (MACD - Signal)
                        df['Histogram'] = df['MACD'] - df['Signal']
                        
                        # Create the figure with two subplots (price and MACD)
                        fig = px.line(title=f"{ticker} - Moving Average Convergence Divergence (MACD)")
                        fig.add_scatter(x=df.index, y=df['MACD'], name='MACD')
                        fig.add_scatter(x=df.index, y=df['Signal'], name='Signal')
                        
                        # Add the histogram as a bar chart
                        fig.add_bar(x=df.index, y=df['Histogram'], name='Histogram')
                        
                        # Add a horizontal line at zero
                        fig.add_hline(y=0, line_dash="dash", line_color="gray")
                        
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error plotting MACD for {ticker}: {e}")
        
    else:
        st.write('Select at least 3 of the 10 tickers to build a MORE EFFECTIVE portfolio')
        st.error("Please select more tickers to continue.")

# AI Stock Advisor tab
with tabs[1]:
    # Sidebar with API key input in this tab
    api_key = get_openai_api_key()
    client = get_openai_client(api_key)
    
    # Main layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Chat with your Stock Advisor")
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask about stocks, investment strategies, or specific companies..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Only proceed if API key is provided
            if not client:
                with st.chat_message("assistant"):
                    st.markdown("Please provide your OpenAI API key in the sidebar to continue.")
                st.session_state.messages.append({"role": "assistant", "content": "Please provide your OpenAI API key in the sidebar to continue."})
            else:
                # Display assistant response
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response = process_query(client, prompt)
                        st.markdown(response)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Right sidebar for visualizations if a ticker has been selected
    with col2:
        if st.session_state.current_ticker and st.session_state.ticker_data:
            ticker = st.session_state.current_ticker
            stock_data = st.session_state.ticker_data
            
            st.subheader(f"{ticker} Insights")
            
            # Stock price chart
            st.plotly_chart(get_stock_price_chart(ticker, stock_data["history"]), use_container_width=True)
            
            # Technical Indicators
            with st.expander("Technical Indicators", expanded=False):
                tech_df = calculate_technical_indicators(stock_data["history"])
                fig = plot_technical_indicators(tech_df, ticker)
                st.pyplot(fig)
            
            # Financial Metrics
            with st.expander("Financial Metrics", expanded=True):
                metrics = get_financial_metrics(stock_data)
                for key, value in metrics.items():
                    if key == "Market Cap" and isinstance(value, (int, float)):
                        value = f"${value/1e9:.2f}B" if value >= 1e9 else f"${value/1e6:.2f}M"
                    elif key == "Dividend Yield" and isinstance(value, (int, float)):
                        value = f"{value*100:.2f}%" if value else 'N/A'
                    st.metric(key, value)
            
            # Recent News
            with st.expander("Recent News", expanded=True):
                news = get_recent_news(ticker)
                for item in news:
                    st.markdown(f"• {item}")

# Hide Streamlit style
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
