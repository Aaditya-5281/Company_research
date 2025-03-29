import os
import streamlit as st
from dotenv import load_dotenv
import time
from pathlib import Path
import shutil

# Import the necessary modules from pyautogen
from autogen import AssistantAgent, config_list_from_json
from autogen.agentchat.contrib.capabilities.teachability import Teachability
import autogen

# Load environment variables
load_dotenv()

# Create outputs directory if it doesn't exist
OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(exist_ok=True)

# Validate environment variables
def validate_env_vars():
    required_vars = ["OPENAI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_SEARCH_ENGINE_ID"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        st.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        st.stop()

# Define the search function
@st.cache_data(ttl=3600)  # Cache results for 1 hour
def google_search(query: str, num_results: int = 2, max_chars: int = 500) -> list:
    import requests
    from bs4 import BeautifulSoup

    api_key = os.getenv("GOOGLE_API_KEY")
    search_engine_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID")

    if not api_key or not search_engine_id:
        raise ValueError("API key or Search Engine ID not found in environment variables")

    url = "https://customsearch.googleapis.com/customsearch/v1"
    params = {
        "key": api_key,
        "cx": search_engine_id,
        "q": query,
        "num": min(num_results, 10)  # Limit to 10 results max
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        st.error(f"Google API request failed: {str(e)}")
        return []

    results = response.json().get("items", [])

    def get_page_content(url: str) -> str:
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")
            text = soup.get_text(separator=" ", strip=True)
            return text[:max_chars]
        except Exception as e:
            print(f"Error fetching {url}: {str(e)}")
            return ""

    enriched_results = []
    for item in results:
        body = get_page_content(item["link"])
        enriched_results.append({
            "title": item["title"],
            "link": item["link"],
            "snippet": item["snippet"],
            "body": body
        })
        time.sleep(1)  # Rate limiting

    return enriched_results

# Define the stock analysis function
@st.cache_data(ttl=1800)  # Cache results for 30 minutes
def analyze_stock(ticker: str) -> dict:
    import yfinance as yf
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from datetime import datetime, timedelta
    from pytz import timezone

    try:
        stock = yf.Ticker(ticker)
        
        # Get historical data
        end_date = datetime.now(timezone("UTC"))
        start_date = end_date - timedelta(days=365)
        hist = stock.history(start=start_date, end=end_date)

        if hist.empty:
            return {"error": f"No historical data available for ticker {ticker}"}

        # Clean up old plot files
        plot_file_path = OUTPUTS_DIR / f"{ticker}_stockprice.png"
        if plot_file_path.exists():
            plot_file_path.unlink()

        # Calculate metrics
        current_price = hist["Close"].iloc[-1]
        year_high = hist["High"].max()
        year_low = hist["Low"].min()
        ma_50 = hist["Close"].rolling(window=50).mean().iloc[-1]
        ma_200 = hist["Close"].rolling(window=200).mean().iloc[-1]

        # Calculate YTD metrics
        ytd_start = datetime(end_date.year, 1, 1, tzinfo=timezone("UTC"))
        ytd_data = hist.loc[ytd_start:]
        if not ytd_data.empty:
            price_change = ytd_data["Close"].iloc[-1] - ytd_data["Close"].iloc[0]
            percent_change = (price_change / ytd_data["Close"].iloc[0]) * 100
        else:
            price_change = percent_change = np.nan

        # Determine trend
        if pd.notna(ma_50) and pd.notna(ma_200):
            trend = "Upward" if ma_50 > ma_200 else "Downward" if ma_50 < ma_200 else "Neutral"
        else:
            trend = "Insufficient data for trend analysis"

        # Calculate volatility
        daily_returns = hist["Close"].pct_change().dropna()
        volatility = daily_returns.std() * np.sqrt(252)  # Annualized volatility

        # Generate plot
        plt.figure(figsize=(12, 6))
        plt.plot(hist.index, hist["Close"], label="Close Price")
        plt.plot(hist.index, hist["Close"].rolling(window=50).mean(), label="50-day MA")
        plt.plot(hist.index, hist["Close"].rolling(window=200).mean(), label="200-day MA")
        plt.title(f"{ticker} Stock Price (Past Year)")
        plt.xlabel("Date")
        plt.ylabel("Price ($)")
        plt.legend()
        plt.grid(True)

        # Save plot
        plt.savefig(plot_file_path)
        plt.close()

        result = {
            "ticker": ticker,
            "current_price": float(current_price),
            "52_week_high": float(year_high),
            "52_week_low": float(year_low),
            "50_day_ma": float(ma_50),
            "200_day_ma": float(ma_200),
            "ytd_price_change": float(price_change),
            "ytd_percent_change": float(percent_change),
            "trend": trend,
            "volatility": float(volatility),
            "plot_file_path": str(plot_file_path)
        }

        return result

    except Exception as e:
        return {"error": f"Error analyzing stock {ticker}: {str(e)}"}

# Streamlit app setup
st.set_page_config(
    page_title="Company Research Agent",
    page_icon="📊",
    layout="wide"
)

# Validate environment variables before running the app
validate_env_vars()

# Configure autogen
@st.cache_resource
def get_config():
    return [{
        "model": "gpt-4-turbo-preview",
        "api_key": os.getenv("OPENAI_API_KEY"),
        "temperature": 0.7
    }]

# Create the agents with proper configurations
def create_agents(config_list):
    llm_config = {
        "config_list": config_list,
        "temperature": 0.7,
        "functions": [
            {
                "name": "google_search",
                "description": "Search Google for information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The search query"},
                        "num_results": {"type": "integer", "description": "Number of results"},
                        "max_chars": {"type": "integer", "description": "Maximum characters"}
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "analyze_stock",
                "description": "Analyze stock data",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ticker": {"type": "string", "description": "Stock ticker symbol"}
                    },
                    "required": ["ticker"]
                }
            }
        ]
    }

    try:
        search_agent = AssistantAgent(
            name="Google_Search_Agent",
            llm_config=llm_config,
            system_message="You are a financial information search specialist. Focus on recent, reliable sources.",
            function_map={"google_search": google_search}
        )

        stock_analysis_agent = AssistantAgent(
            name="Stock_Analysis_Agent",
            llm_config=llm_config,
            system_message="You are a stock analysis expert. Provide clear, actionable insights.",
            function_map={"analyze_stock": analyze_stock}
        )

        report_agent = AssistantAgent(
            name="Report_Agent",
            llm_config={"config_list": config_list},
            system_message="You are a financial report writer. Create comprehensive, well-structured reports."
        )

        return search_agent, stock_analysis_agent, report_agent

    except Exception as e:
        st.error(f"Failed to create agents: {str(e)}")
        return None, None, None

# Main app interface
st.title("Company Research Agent")
st.subheader("Financial Analysis and Report Generator")

with st.expander("About this app", expanded=True):
    st.markdown("""
    This app uses AI agents to research companies and generate comprehensive financial reports:
    
    1. **Search Agent**: Finds latest company information
    2. **Stock Analysis Agent**: Analyzes stock performance
    3. **Report Agent**: Compiles data into a detailed report
    
    Enter a company name below to begin!
    """)

# Input section
company_name = st.text_input("Enter company name:", value="")
ticker = st.text_input("Enter stock ticker (optional):", value="")

# Auto-suggest ticker
if not ticker and company_name:
    common_tickers = {
        "american airlines": "AAL",
        "apple": "AAPL",
        "microsoft": "MSFT",
        "amazon": "AMZN",
        "google": "GOOGL",
        "alphabet": "GOOGL",
        "meta": "META",
        "facebook": "META",
        "tesla": "TSLA",
        "nvidia": "NVDA"
    }
    suggested_ticker = common_tickers.get(company_name.lower())
    if suggested_ticker:
        ticker = suggested_ticker
        st.info(f"Using suggested ticker: {ticker}")

# Generate report button
if st.button("Generate Report"):
    if not company_name:
        st.error("Please enter a company name.")
    else:
        try:
            with st.spinner("Initializing AI agents..."):
                config_list = get_config()
                search_agent, stock_analysis_agent, report_agent = create_agents(config_list)

                if not all([search_agent, stock_analysis_agent, report_agent]):
                    st.error("Failed to initialize agents. Please try again.")
                    st.stop()

            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Output containers
            search_output = st.expander("Search Results", expanded=False)
            stock_output = st.expander("Stock Analysis", expanded=False)
            report_output = st.expander("Final Report", expanded=True)

            # Replace the report generation section in your code with this:

            # Stock Analysis
            status_text.text("Analyzing stock data...")
            progress_bar.progress(0.2)
            
            stock_analysis_data = None
            if ticker:
                stock_message = f"Analyze the stock for {company_name} ({ticker})"
                stock_response = stock_analysis_agent.generate_reply(
                    messages=[{"role": "user", "content": stock_message}]
                )
                
                # Extract the actual stock analysis results
                if isinstance(stock_response, dict) and 'function_call' in stock_response:
                    # Execute the stock analysis
                    stock_analysis_data = analyze_stock(ticker)
                    stock_output.markdown("### Stock Analysis Results")
                    for key, value in stock_analysis_data.items():
                        if key != 'plot_file_path':
                            stock_output.write(f"**{key}:** {value}")
                else:
                    stock_output.markdown(stock_response)

            # Web Search
            status_text.text("Gathering company information...")
            progress_bar.progress(0.5)
            
            search_message = f"Find recent financial news and information about {company_name}"
            search_response = search_agent.generate_reply(
                messages=[{"role": "user", "content": search_message}]
            )
            
            # Extract the actual search results
            search_results = None
            if isinstance(search_response, dict) and 'function_call' in search_response:
                # Execute the search
                search_args = eval(search_response['function_call']['arguments'])
                search_results = google_search(**search_args)
                search_output.markdown("### Search Results")
                for result in search_results:
                    search_output.markdown(f"""
                    **{result['title']}**  
                    {result['snippet']}  
                    [Link]({result['link']})
                    """)
            else:
                search_output.markdown(search_response)

            # Generate Report
            status_text.text("Compiling final report...")
            progress_bar.progress(0.8)
            
            # Prepare the report prompt with actual data
            report_sections = [
                f"Company: {company_name}",
                "\nStock Analysis:" if stock_analysis_data else "",
                str(stock_analysis_data) if stock_analysis_data else "",
                "\nRecent News and Information:",
                "\n".join([f"- {result['title']}: {result['snippet']}" 
                          for result in search_results]) if search_results else str(search_response)
            ]
            
            report_prompt = f"""
            Generate a comprehensive financial report for {company_name} using the following information:

            {'\n'.join(report_sections)}

            Format the report with these sections:
            1. Executive Summary
            2. Company Overview
            3. Recent News and Developments
            4. {'Stock Analysis and Performance' if ticker else 'Market Position'}
            5. Industry Analysis
            6. Future Outlook
            7. Conclusion

            End with 'REPORT COMPLETE'
            """
            
            report_response = report_agent.generate_reply(
                messages=[{"role": "user", "content": report_prompt}]
            )
            
            # Display report
            clean_report = report_response.replace("REPORT COMPLETE", "").strip() if isinstance(report_response, str) else str(report_response)
            report_output.markdown(clean_report)

            # Show stock chart if available
            if ticker and stock_analysis_data and 'plot_file_path' in stock_analysis_data:
                plot_path = Path(stock_analysis_data['plot_file_path'])
                if plot_path.exists():
                    st.subheader(f"{ticker} Stock Price Chart")
                    st.image(str(plot_path))

            # Download options
            st.download_button(
                label="Download Report (TXT)",
                data=clean_report,
                file_name=f"{company_name}_report.txt",
                mime="text/plain"
            )

            # Complete
            progress_bar.progress(1.0)
            status_text.text("Report generation complete!")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            
# Footer
st.markdown("---")
st.caption("Powered by AutoGen and OpenAI GPT-4")

# Cleanup on session end
if st.session_state.get('ended'):
    shutil.rmtree(OUTPUTS_DIR, ignore_errors=True)