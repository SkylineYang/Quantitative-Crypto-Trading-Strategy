import numpy as np
import pandas as pd
import pandas_ta as ta
from typing import Tuple
from dateutil.relativedelta import relativedelta

def getTechnicalIndicators(
        filename: str,
        window: int = 14
    ) -> pd.DataFrame:
    '''
    Get historical price data with some technical indicators as features in feature engineering process. \n
    The technical indicators are: SMA, RSI, MACD, BB, SO and ATR. \n
    Default window days for RSI, BB, SO and ATR: 14 days.
    '''
    data = pd.read_excel(filename, index_col=0)
    data.index = pd.to_datetime(data.index)

    ### Simple Moving Average
    data["SMA_7"] = ta.sma(data["Price"], length=7)
    data["SMA_30"] = ta.sma(data["Price"], length=30)

    ### Relative Strength Index
    data[f"RSI_{window}"] = ta.rsi(data["Price"], length=window)

    ### Moving Average Covergence/Divergence
    macd = ta.macd(data["Price"], fast=12, slow=26, signal=9)
    data["MACD"] = macd["MACD_12_26_9"]
    data["MACD_Hist"] = macd["MACDh_12_26_9"]

    ### Bollinger Bands
    bb = ta.bbands(data["Price"], length=window, std=2)
    data["BB_upper"] = bb[f"BBU_{window}_2.0"]
    data["BB_middle"] = bb[f"BBM_{window}_2.0"]
    data["BB_lower"] = bb[f"BBL_{window}_2.0"]

    ### Stochastic Osillator
    stoch = ta.stoch(data["Price"], data["Price"], data["Price"], k=window, d=3)
    data["Stoch_K"] = stoch[f"STOCHk_{window}_3_3"]
    data["Stoch_D"] = stoch[f"STOCHd_{window}_3_3"]

    ### Average True Range
    data[f"ATR_{window}"] = ta.atr(high=data["Price"], low=data["Price"], close=data["Price"], length=window)

    return data

def movingWindow(
        data: pd.DataFrame,
        window_size: int = 14
    ) -> pd.DataFrame:
    '''
    Use moving window method to split historical crypto price and its features into train and test dataset. \n
    This function returns X_train, X_test, y_train and y_test, where X represents the past 14 days' prices and technical indicators, while y represents the 15th day's price.
    '''

    data["Date"] = pd.to_datetime(data.index)
    feature_columns = [col for col in data.columns if col != "Date"]

    X_records = []
    X_dates = []

    y_records = []
    y_dates = []

    for i in range(len(data) - window_size):
        window_data = data.iloc[i:i + window_size]
        
        record = {}
        for day_offset in range(window_size):
            for col in feature_columns:
                record[f"{col}_t-{window_size - day_offset}"] = window_data.iloc[day_offset][col]
        
        X_records.append(record)
        X_dates.append(data.iloc[i + window_size]["Date"])
        
        y_records.append(data.iloc[i + window_size]["Price"])
        y_dates.append(data.iloc[i + window_size]["Date"])

    X_df = pd.DataFrame(X_records)
    X_df["Date"] = X_dates
    X_df = X_df.set_index("Date")
    X_df = X_df.dropna()

    y_df = pd.DataFrame(y_records, columns=["Price"])
    y_df["Date"] = y_dates
    y_df = y_df.set_index("Date")
    y_df = y_df.loc[X_df.index]
    
    return X_df, y_df

def traintestSplit(
        X_df: pd.DataFrame, 
        y_df: pd.DataFrame,
        stride: int = 15,
        interval: int = 30,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    '''
    Split train and test set on preprocessed dataframes.
    '''
    X_train, X_test, y_train, y_test = [], [], [], []

    startdate = X_df.index[0]

    while startdate <= X_df.index[-1]:
        enddate = min(startdate + relativedelta(days=interval), X_df.index[-1])
        splitdate = startdate + 0.8 * (enddate-startdate)
        X_train.append(X_df[(X_df.index >= startdate) & (X_df.index < splitdate)].values)
        X_test.append(X_df[(X_df.index >= splitdate) & (X_df.index < enddate)].values)
        y_train.append(y_df[(y_df.index >= startdate) & (y_df.index < splitdate)].values)
        y_test.append(y_df[(y_df.index >= splitdate) & (y_df.index < enddate)].values)
        startdate += relativedelta(days=stride)

    X_train = np.array(X_train, dtype=object)
    X_test = np.array(X_test, dtype=object)
    y_train = np.array(y_train, dtype=object)
    y_test = np.array(y_test, dtype=object)

    return X_train, X_test, y_train, y_test