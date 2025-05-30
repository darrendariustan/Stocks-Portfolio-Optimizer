# 📈 Stock Portfolio Optimizer & AI Stock Advisor

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)
![OpenAI](https://img.shields.io/badge/AI-GPT--4o-green.svg)

## 🔗 [Try the Live Demo with your own API Key](https://stocks-portfolio-optimizer.onrender.com/)

An advanced application that combines modern portfolio theory with artificial intelligence to optimize your investment strategy and provide real-time stock analysis.

## ✨ Key Features

### 🤖 AI-Powered Stock Analysis
- **GPT-4o Integration**: Leverages OpenAI's powerful language model to analyze stocks and provide insights
- **Natural Language Processing**: Ask questions about any stock and get intelligent, contextualized responses
- **News Sentiment Analysis**: Automatically aggregates and analyzes recent news about your selected stocks

### 📊 Advanced Technical Analysis
- **Interactive Charts**: Beautiful visualizations of stock price history and performance metrics
- **Technical Indicators**: 
  - Moving Averages (SMA, EMA)
  - Relative Strength Index (RSI)
  - Moving Average Convergence Divergence (MACD)
  - Bollinger Bands
- **Candlestick Patterns**: Professional-grade candlestick charts with zoom capabilities

### 💼 Portfolio Optimization
- **Efficient Frontier Analysis**: Optimize your portfolio based on the Nobel Prize-winning Modern Portfolio Theory
- **Risk-Return Optimization**: Find the optimal balance between expected returns and volatility
- **Maximum Sharpe Ratio**: Identify the portfolio allocation with the best risk-adjusted returns
- **Multiple Optimization Methods**: Choose between various optimization strategies

### 📱 Modern User Interface
- **Multi-tab Design**: Easily navigate between different features and analysis tools
- **Interactive Elements**: Adjust parameters in real-time to see how they affect your portfolio
- **Responsive Design**: Works on desktop and mobile devices

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/darrendariustan/Stocks-Portfolio-Optimizer.git

# Navigate to the project directory
cd Stocks-Portfolio-Optimizer

# Install required dependencies
pip install -r requirements.txt
```

## 🔍 Usage

```bash
# Run the Streamlit app
streamlit run finapp_final.py
```

## �� Application Demo

Video demonstrations of the application are available in the repository:
- [PDAI_LLM2_Demo.webm](PDAI_LLM2_Demo.webm) - PDAI LLM Project 2 demonstration

Watch the demos to see all the features in action without needing to set up the environment locally.

## 🧠 AI Features

To utilize the AI features, you'll need an OpenAI API key. Enter your key in the sidebar when prompted.

## 📚 Technology Stack

- **Streamlit**: For the interactive web application
- **PyPortfolioOpt**: For portfolio optimization algorithms
- **Yahoo Finance API**: For real-time and historical stock data
- **Plotly & Matplotlib**: For interactive data visualization
- **OpenAI API**: For AI-powered stock analysis and recommendations
- **Pandas & NumPy**: For data manipulation and analysis

## Render Deployment Instructions

When deploying this application on Render, follow these steps:

1. Connect your GitHub repository to Render
2. Create a new Web Service 
3. Use the following build command to ensure all dependencies are properly installed and the PyPortfolioOpt patch is applied:
   ```
   pip install -r requirements.txt && python patch_pypfopt.py
   ```
4. Set the start command to:
   ```
   streamlit run finapp_final.py
   ```
5. Add your OpenAI API key as an environment variable named `OPENAI_API_KEY`

The `patch_pypfopt.py` script automatically fixes an issue with the PyPortfolioOpt library that can cause errors with the matplotlib "seaborn-deep" style on Render.
