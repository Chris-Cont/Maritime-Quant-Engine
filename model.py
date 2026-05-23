import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def train_and_backtest(df):
    """Trains the Random Forest model and executes out-of-sample backtesting."""
    print("🧠 Training Production Random Forest Model...")
    df = df.copy()
    
    # Define Target: Next day's direction (1 for UP, 0 for DOWN)
    df['Target'] = np.where(df['Ship_Return'].shift(-1) > 0, 1, 0)
    df_clean = df.dropna().copy()

    features = [
        'Ship_Return', 'Oil_Return', 'USD_Return', 'Correlation_30d', 
        'RSI_14', 'MACD_Hist', 'Volatility_7d', 
        'Ship_Ret_Lag1', 'Ship_Ret_Lag2'
    ]
    
    X = df_clean[features]
    y = df_clean['Target']

    # Chronological Split (No Shuffling to prevent look-ahead bias)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Initialize and fit the model
    model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    # VECTORIZED BACKTESTING on unseen test data
    df_test = df_clean.loc[X_test.index].copy()
    df_test['AI_Prediction'] = model.predict(X_test)
    df_test['AI_Signal'] = df_test['AI_Prediction'].shift(1) # Signal acted upon next day
    df_test['AI_Strategy_Return'] = df_test['AI_Signal'] * df_test['Ship_Return']

    # Capital tracking ($10,000 Initial)
    initial_capital = 10000
    df_test['Buy_and_Hold_Bal'] = initial_capital * (1 + df_test['Ship_Return']).cumprod()
    df_test['AI_Strategy_Bal'] = initial_capital * (1 + df_test['AI_Strategy_Return']).cumprod()

    # Predict tomorrow's directional movement using today's data
    today_data = df_clean[features].iloc[-1:]
    tomorrow_prediction = model.predict(today_data)[0]

    return df_test, tomorrow_prediction, model.feature_importances_, features
