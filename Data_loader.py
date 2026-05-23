import yfinance as yf
import pandas as pd

def fetch_maritime_data(period="10y"):
    """Fetches 10 years of historical data for Shipping, Commodities, and Macro Forex."""
    print(f"📥 Fetching {period} historical data from markets...")
    
    df_ship = yf.download('FRO', period=period, progress=False)['Close']
    df_oil = yf.download('CL=F', period=period, progress=False)['Close']
    macro_data = yf.download('USDEUR=X', period=period, progress=False)['Close']

    # Handle multi-index columns if returned by yfinance
    if isinstance(df_ship, pd.DataFrame): df_ship = df_ship.squeeze()
    if isinstance(df_oil, pd.DataFrame): df_oil = df_oil.squeeze()
    if isinstance(macro_data, pd.DataFrame): macro_data = macro_data.squeeze()

    # Merge into a master dataframe
    df_master = pd.DataFrame({
        'Ship_Price': df_ship, 
        'Oil_Price': df_oil, 
        'USD_EUR_Rate': macro_data
    })
    
    # Forward fill gaps (weekends/holidays) and drop initial NaNs
    df_master = df_master.ffill().dropna()
    return df_master
