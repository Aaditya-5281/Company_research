# Company Research Agent

A powerful AI-powered financial research tool that generates comprehensive company reports using multiple specialized AI agents. This application combines web search capabilities, stock analysis, and AI-driven report generation to provide detailed insights about companies.

## Features

- **Multi-Agent System**: Utilizes three specialized AI agents:
  - Google Search Agent: Gathers latest company information
  - Stock Analysis Agent: Analyzes stock performance and trends
  - Report Agent: Compiles data into comprehensive reports

- **Stock Analysis**:
  - Historical price data visualization
  - Technical indicators (50-day and 200-day moving averages)
  - Volatility analysis
  - YTD performance metrics
  - Interactive stock charts

- **Web Research**:
  - Google Custom Search integration
  - Recent news and information gathering
  - Source verification and credibility assessment

- **Report Generation**:
  - Executive summary
  - Company overview
  - Recent developments
  - Stock analysis (when applicable)
  - Industry analysis
  - Future outlook
  - Conclusion

## Prerequisites

- Python 3.7+
- OpenAI API key
- Google Custom Search API key
- Google Search Engine ID

## Environment Variables

Create a `.env` file in the project root with the following variables:

```env
OPENAI_API_KEY=your_openai_api_key
GOOGLE_API_KEY=your_google_api_key
GOOGLE_SEARCH_ENGINE_ID=your_search_engine_id
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/company-research-agent.git
cd company-research-agent
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the provided local URL (typically http://localhost:8501)

3. Enter a company name and optional stock ticker symbol

4. Click "Generate Report" to start the analysis

5. View the generated report and download it as a text file if needed

## Dependencies

- streamlit
- python-dotenv
- autogen
- yfinance
- matplotlib
- pandas
- numpy
- requests
- beautifulsoup4

## Project Structure

```
company-research-agent/
├── app.py              # Main application file
├── requirements.txt    # Project dependencies
├── .env               # Environment variables
└── outputs/           # Generated reports and charts
```

## Features in Detail

### Stock Analysis
- Historical price data for the past year
- Moving averages (50-day and 200-day)
- Volatility calculations
- YTD performance metrics
- Interactive stock charts

### Web Research
- Google Custom Search integration
- Recent news and information gathering
- Source verification
- Content extraction and summarization

### Report Generation
- Comprehensive financial analysis
- Market position assessment
- Industry context
- Future outlook
- Downloadable reports in text format

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with [AutoGen](https://github.com/microsoft/autogen)
- Powered by OpenAI GPT-4
- Uses Google Custom Search API
- Stock data provided by Yahoo Finance 
