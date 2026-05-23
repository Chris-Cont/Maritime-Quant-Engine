import pandas as pd
import numpy as np

def build_features(df):
    """Computes mathematical indicators, momentum, volatility, and market memory."""
    print("⚙️ Engineering advanced features and indicators...")
    df = df.copy()
    
    # 1. Base Returns
    df['Ship_Return'] = df['Ship_Price'].pct_change()
    df['Oil_Return'] = df['Oil_Price'].pct_change()
    df['USD_Return'] = df['USD_EUR_Rate'].pct_change()
    
    # 2. Macro Correlation
    df['Correlation_30d'] = df['Ship_Return'].rolling(window=30).corr(df['Oil_Return'])

    # 3. Momentum: RSI (14)
    delta = df['Ship_Price'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))

    # 4. Momentum: MACD Histogram
    ema_12 = df['Ship_Price'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Ship_Price'].ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    signal = macd.ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = macd - signal

    # 5. Volatility & Market Memory (Lags)
    df['Volatility_7d'] = df['Ship_Return'].rolling(window=7).std()
    df['Ship_Ret_Lag1'] = df['Ship_Return'].shift(1)
    df['Ship_Ret_Lag2'] = df['Ship_Return'].shift(2)

    return df.dropna()
