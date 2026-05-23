# Maritime-Quant-Engine
Predictive ML Model for shipping equities

📌 Project Overview
This project is an end-to-end Machine Learning pipeline designed to predict the daily directional movement of shipping equities (e.g., Frontline - FRO). It leverages the fundamental dynamics of the maritime industry by fusing shipping equity prices with critical macroeconomic and commodity data: Crude Oil Futures (bunker fuel costs) and USD/EUR Exchange Rates (global trade currency)
.
🧠 Methodology & Feature Engineering
The model moves beyond traditional technical analysis by utilizing a Random Forest Classifier trained on 10 years of daily market data. The feature engineering pipeline includes:
•	Macro-Commodity Integration: Daily returns of CL=F (Oil) and USDEUR=X (Forex).
•	Dynamic Relationships: 30-day Rolling Correlation between Shipping Equities and Crude Oil.
•	Momentum & Volatility: RSI (14), MACD Histogram, and 7-day Rolling Volatility.
•	Market Memory: Lagged returns to capture short-term market momentum.

📊 Backtesting Results (Out-of-Sample)
The model was evaluated using a strict chronological Train/Test split (no look-ahead bias). The backtest simulates trading on the unseen 20% of the 10-year dataset with an initial capital of $10,000.
•	Model Accuracy: 54.46% (Demonstrating a clear statistical edge / alpha over a daily timeframe).
•	Buy & Hold Final Balance: $14,796.13
•	AI Quant Strategy Final Balance: $27,481.64
•	Performance: The model significantly outperformed the benchmark by successfully predicting market downturns and shifting to a neutral risk position (cash).
🔍 Feature Importance Findings
The Random Forest model identified Crude Oil Returns (12.1%), 30-day Rolling Correlation (11.5%), and 7-day Volatility (11.2%) as the most critical predictors, completely overshadowing traditional technical indicators like RSI and MACD. This validates the hypothesis that shipping equities are primarily driven by macro-commodity cycles.

💻 Tech Stack
•	Data Pipelines: yfinance, pandas, numpy
•	Machine Learning: scikit-learn (Random Forest Classifier)
•	Data Visualization: matplotlib
