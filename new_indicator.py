import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def sigmoid(x):
    constant = 2
    return 2*((1/(1+constant**(-x)))-0.5)

def generatorTrendList(stock_data):
    # Preprocessed on the historical data
    candles = np.array(stock_data['Close']-stock_data['Open'])
    midpoint = 0.5*(stock_data['Close']+stock_data['Open'])
    hl_midpoint = 0.5*(stock_data['High']+stock_data['Low'])
    points = 0.5*midpoint + 0.5*hl_midpoint

    # some initializations 
    candleWindow = 3 # makes observation on three previous candles
    len_candles = len(candles)
    trendList = np.zeros(len_candles)
    meanPoints = np.zeros(len_candles)
    meanList = np.zeros(len_candles)
    accumulations = np.zeros(len_candles)

    for i in range(candleWindow+1, len_candles):

        # trend boost : if a trend persists for 3 days, then highly likely the same trend will continue
        correction1 = 0
        if all(candles[j] > 0 for j in range(i-candleWindow, i+1)):
            correction1 = 1.61
        elif all(candles[j] < 0 for j in range(i-candleWindow, i+1)):
            correction1 = 1.61
        else:
            correction1 = 0.8

        # Yet to be implemented: correction 2 would be based on the changing candlelight and change in area of candle to predict a turnover
        correction2 = 0

        
        weights = np.array([0.05, 0.05, 0.23, 0.67])
        meanList[i] = np.dot(candles[i-candleWindow-1:i], weights) # valuation of the stock just the day before
        meanPoints[i] = np.dot(weights, points[i-candleWindow:i+1])
        accumulation = (points[i] - meanPoints[i-1])*correction1 # deviation from the previous price multiplied by correction term
        accumulations[i] = accumulation
        trendList[i] = sigmoid(accumulation) # to regularize the output
        
        # from what I have observed, trend values between -0.7 to 0.7 signify a sideways trend
        #   and values beyond +-0.85 signify very good trend.

    return trendList
    


if __name__ == '__main__':
    ticker_symbol = 'FORTIS.NS'
    start_date = datetime(2022, 2, 1)  # YYYY, MM, DD

    # ~ toggle the comment for the below command to have a custome end_date
    end_date = datetime(2022, 4, 10)  # YYYY, MM, DD 

    # # ~ toggle the comment for the below command to have an end_date time some delta time
    # timeframe = 150 # how many days ahead from the start date do you want the data
    # end_date = start_date + timedelta(days=timeframe)

    end_date = end_date.strftime("%Y-%m-%d")
    start_date = start_date.strftime("%Y-%m-%d")
    stock_data = yf.download(ticker_symbol, start = start_date, end=end_date, interval="1d")
    trends = generatorTrendList(stock_data)
    stock_data['trend'] = trends
    print(stock_data[['trend']])

