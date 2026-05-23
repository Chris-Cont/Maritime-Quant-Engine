from data_loader import fetch_maritime_data
from features import build_features
from model import train_and_backtest

if __name__ == "__main__":
    print("🚢 MARITIME QUANT ENGINE - PRODUCTION INITIALIZED 🚢\n" + "="*55)
    
    # 1. Pipeline Execution
    raw_data = fetch_maritime_data(period="10y")
    featured_data = build_features(raw_data)
    backtest_results, tomorrow_signal, feature_imps, feature_names = train_and_backtest(featured_data)
    
    # 2. Performance Metrics
    final_bh = backtest_results['Buy_and_Hold_Bal'].iloc[-1]
    final_ai = backtest_results['AI_Strategy_Bal'].iloc[-1]
    
    print("\n" + "="*55 + "\n📊 BACKTEST PERFORMANCE METRICS\n" + "="*55)
    print(f"📉 Benchmark Strategy (Buy & Hold) : ${final_bh:,.2f}")
    print(f"🤖 Production Quant AI Strategy   : ${final_ai:,.2f}")
    print(f"💰 Net Alpha Generated            : ${final_ai - final_bh:,.2f}")
    
    # 3. Actionable Signal Output
    print("\n" + "="*55 + "\n🚀 LIVE TRADING SIGNAL FOR TOMORROW\n" + "="*55)
    if tomorrow_signal == 1:
        print(" DIRECTIONAL FORECAST: [ BUY / LONG ] - Equity expected to close UP.")
    else:
        print(" DIRECTIONAL FORECAST: [ LIQUIDATE / CASH ] - Equity expected to close DOWN.")
    print("="*55)
