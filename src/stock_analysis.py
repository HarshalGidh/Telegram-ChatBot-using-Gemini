from flask import Flask, request, jsonify
import yfinance as yf
import pandas as pd
import requests
import os
import logging

app = Flask(__name__)

NEWS_API_KEY = os.getenv('NEWS_API_KEY')

@app.route('/fetch_stock_data', methods=['GET'])
def fetch_stock_data():
    ticker = request.args.get('ticker')
    if not ticker:
        return jsonify({"error": "Ticker is required"}), 400

    stock = yf.Ticker(ticker)
    data = {}

    company_details = stock.info.get('longBusinessSummary', 'No details available')
    data['Company Details'] = company_details
    sector = stock.info.get('sector', 'No sector information available')
    data['Sector'] = sector
    prev_close = stock.info.get('previousClose', 'No previous close price available')
    data['Previous Closing Price'] = prev_close
    open_price = stock.info.get('open', 'No opening price available')
    data['Today Opening Price'] = open_price

    hist = stock.history(period="5d")
    if not hist.empty and 'Close' in hist.columns:
        if hist.index[-1].date() == yf.download(ticker, period="1d").index[-1].date():
            close_price = hist['Close'].iloc[-1]
            data['Todays Closing Price'] = close_price
        else:
            data['Todays Closing Price'] = "Market is open, there is no closing price available yet."
    else:
        data['Todays Closing Price'] = "No historical data available for closing price."

    day_high = stock.info.get('dayHigh', 'No high price available')
    data['Today High Price'] = day_high
    day_low = stock.info.get('dayLow', 'No low price available')
    data['Today Low Price'] = day_low
    volume = stock.info.get('volume', 'No volume information available')
    data['Today Volume'] = volume
    dividends = stock.info.get('dividendRate', 'No dividend information available')
    data['Today Dividends'] = dividends
    splits = stock.info.get('lastSplitFactor', 'No stock split information available')
    data['Today Stock Splits'] = splits
    pe_ratio = stock.info.get('trailingPE', 'No P/E ratio available')
    data['P/E Ratio'] = pe_ratio
    market_cap = stock.info.get('marketCap', 'No market cap available')
    data['Market Cap'] = market_cap

    income_statement = stock.financials
    balance_sheet = stock.balance_sheet
    cashflow = stock.cashflow

    news_url = f'https://newsapi.org/v2/everything?q={ticker}&apiKey={NEWS_API_KEY}&pageSize=3'
    news_response = requests.get(news_url)
    if news_response.status_code == 200:
        news_data = news_response.json()
        articles = news_data.get('articles', [])
        if articles:
            top_news = "\n\n".join([f"{i+1}. {article['title']} - {article['url']}" for i, article in enumerate(articles)])
            data['Top News'] = top_news
        else:
            data['Top News'] = "No news articles found."
    else:
        data['Top News'] = "Failed to fetch news articles."

    graph_url = f"https://finance.yahoo.com/chart/{ticker}"
    data['graph_url'] = graph_url

    file_path = os.path.join('data', f'{ticker}_financial_data.xlsx')
    with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
        income_statement.to_excel(writer, sheet_name='Income Statement')
        balance_sheet.to_excel(writer, sheet_name='Balance Sheet')
        cashflow.to_excel(writer, sheet_name='Cashflow')

    data_list = list(data.items())
    data_str = str(data_list)

    return jsonify({
        "data": data,
        "file_path": file_path,
        "data_str": data_str
    })

@app.route('/analyze_stock_data', methods=['GET'])
def analyze_stock_data():
    ticker = request.args.get('ticker')
    if not ticker:
        return jsonify({"error": "Ticker is required"}), 400

    hist, data_str, file_path = fetch_stock_data(ticker)
    avg_close = hist['Close'].mean()
    formatted_data = extract_excel_data(file_path)

    task = f"""You are a Stock Market Expert. You know everything about stock market trends and patterns.
                Based on the provided stock data, analyze the stock's performance, including whether it is overvalued or undervalued.
                Predict the stock price range for the next week and provide reasons for your prediction.
                Advise whether to buy this stock now or not, with reasons for your advice."""

    query = task + "\nStock Data: " + data_str + "\nFinancial Data: " + formatted_data
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(query)
    
    # Log the response object to understand its structure
    logging.info(f"Model response: {response}")
    
    # Extract the text content from the response
    try:
        response_text = response.text
        format_response = markdown_to_text(response_text)
    except Exception as e:
        logging.error(f"Error extracting text from response: {e}")
        return jsonify({"error": "Failed to analyze stock data"}), 500

    return jsonify({
        "average_closing_price": f"${avg_close:.2f}",
        "analysis": format_response
    })

def extract_excel_data(file_path):
    financial_data = ""
    xls = pd.ExcelFile(file_path)
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name)
        financial_data += f"\n\nSheet: {sheet_name}\n"
        financial_data += df.to_string()
    return financial_data

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)







# import yfinance as yf
# import pandas as pd
# import requests
# from aiogram import types
# from aiogram.types import InputFile
# import os
# import logging

# NEWS_API_KEY = os.getenv('NEWS_API_KEY')

# async def fetch_stock_data(message, ticker):
#     stock = yf.Ticker(ticker)
#     chat_id = message.chat.id
#     data = {}

#     company_details = stock.info.get('longBusinessSummary', 'No details available')
#     data['Company Details'] = company_details
#     sector = stock.info.get('sector', 'No sector information available')
#     data['Sector'] = sector
#     prev_close = stock.info.get('previousClose', 'No previous close price available')
#     data['Previous Closing Price'] = prev_close
#     open_price = stock.info.get('open', 'No opening price available')
#     data['Today Opening Price'] = open_price

#     hist = stock.history(period="5d")
#     if not hist.empty and 'Close' in hist.columns:
#         if hist.index[-1].date() == yf.download(ticker, period="1d").index[-1].date():
#             close_price = hist['Close'].iloc[-1]
#             data['Todays Closing Price'] = close_price
#         else:
#             data['Todays Closing Price'] = "Market is open, there is no closing price available yet."
#     else:
#         data['Todays Closing Price'] = "No historical data available for closing price."

#     day_high = stock.info.get('dayHigh', 'No high price available')
#     data['Today High Price'] = day_high
#     day_low = stock.info.get('dayLow', 'No low price available')
#     data['Today Low Price'] = day_low
#     volume = stock.info.get('volume', 'No volume information available')
#     data['Today Volume'] = volume
#     dividends = stock.info.get('dividendRate', 'No dividend information available')
#     data['Today Dividends'] = dividends
#     splits = stock.info.get('lastSplitFactor', 'No stock split information available')
#     data['Today Stock Splits'] = splits
#     pe_ratio = stock.info.get('trailingPE', 'No P/E ratio available')
#     data['P/E Ratio'] = pe_ratio
#     market_cap = stock.info.get('marketCap', 'No market cap available')
#     data['Market Cap'] = market_cap

#     income_statement = stock.financials
#     balance_sheet = stock.balance_sheet
#     cashflow = stock.cashflow

#     news_url = f'https://newsapi.org/v2/everything?q={ticker}&apiKey={NEWS_API_KEY}&pageSize=3'
#     news_response = requests.get(news_url)
#     if news_response.status_code == 200:
#         news_data = news_response.json()
#         articles = news_data.get('articles', [])
#         if articles:
#             top_news = "\n\n".join([f"{i+1}. {article['title']} - {article['url']}" for i, article in enumerate(articles)])
#             data['Top News'] = top_news
#         else:
#             data['Top News'] = "No news articles found."
#     else:
#         data['Top News'] = "Failed to fetch news articles."

#     for key, value in data.items():
#         await message.reply(f"{key}: {value}")

#     graph_url = f"https://finance.yahoo.com/chart/{ticker}"
#     data['graph_url'] = graph_url
#     await message.reply(f"Stock Chart: \n{graph_url}")

#     file_path = os.path.join('data', f'{ticker}_financial_data.xlsx')
#     with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
#         income_statement.to_excel(writer, sheet_name='Income Statement')
#         balance_sheet.to_excel(writer, sheet_name='Balance Sheet')
#         cashflow.to_excel(writer, sheet_name='Cashflow')

#     excel_file = FSInputFile(file_path, filename=f'{ticker}_financial_data.xlsx')
#     await bot.send_document(chat_id, excel_file, caption=f'{ticker} Financial Data')

#     data_list = list(data.items())
#     data_str = str(data_list)

#     return hist, data_str, file_path


# async def analyze_stock_data(hist, data, excel_file,chat_id):
#     avg_close = hist['Close'].mean()
#     formatted_data = extract_excel_data(excel_file)

#     task = f"""You are a Stock Market Expert. You know everything about stock market trends and patterns.
#                 Based on the provided stock data, analyze the stock's performance, including whether it is overvalued or undervalued.
#                 Predict the stock price range for the next week and provide reasons for your prediction.
#                 Advise whether to buy this stock now or not, with reasons for your advice."""

#     query = task + "\nStock Data: " + data + "\nFinancial Data: " + formatted_data
#     model = genai.GenerativeModel('gemini-1.5-flash')
#     response = model.generate_content(query)
    
#     # Log the response object to understand its structure
#     logging.info(f"Model response: {response}")
    
#      # Extract the text content from the response
#     try:
#         response_text = response.text #result.candidates[0].content.parts[0].text  #response.candidates[0]['content']['parts'][0]['text']
#         format_response = markdown_to_text(response_text)
#     except Exception as e:
#         logging.error(f"Error extracting text from response: {e}")
#         return f"Average closing price for the last 3 months: ${avg_close:.2f}", response
    

#     # Store the response in chat history
#     write_chat_history(chat_id, {'role': 'bot', 'message': format_response})

#     return f"Average closing price for the last 3 months: ${avg_close:.2f}", format_response


# def extract_excel_data(file_path):
#     financial_data = ""
#     xls = pd.ExcelFile(file_path)
#     for sheet_name in xls.sheet_names:
#         df = pd.read_excel(xls, sheet_name=sheet_name)
#         financial_data += f"\n\nSheet: {sheet_name}\n"
#         financial_data += df.to_string()
#     return financial_data
