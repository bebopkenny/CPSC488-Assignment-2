## Core Performance
- Final balance: 1.1356
- Total gain/loss: 0.1403
- Avg annual return (%): 56.6724
- Total return (%): -0.0038

## Trading Simulation
- trades: 51
- win rate: 0.608 (61%)
- avg PnL per trade: 0.002581
- Median PnL per trade: 0.001594
- equity start: 1.00
- equity end: 1.14
- max drawdown: -1.845%
- By type: Buy 50 trades, mean PnL 0.002581, total PnL 0.129041

## Impact Score Behavior
- Impact score and next day return: -0.1049
- Decile 0: n=39, mean 0.003076, median 0.001758
- Decile 1: n=1, mean 0.008597
- Decile 2: n=5, mean −0.001098, median −0.003823
- Decile 3: n=5, mean 0.001194, median −0.002802

# Vectorization Coverage (news features)
- DTM: rows 0, symbols 0, date range N/A
- TF-IDF: rows 0, symbols 0, date range N/A
- Curated: rows 0, symbols 0, date range N/A

# Notes
1. Total return inconsistency:
- final_balance = 1.1356 implies ~+13.56% cumulative.
- total_return_pct = −0.0038 (−0.38%) conflicts.
- Recommendation: In the write up, treat final_balance and the equity curve as the ground-truth outcome and flag total_return_pct as a calculation artifact to be corrected.
2. S&P/market adjusted NaNs:
- The market model fields are NaN, so alpha/beta/abnormal returns weren’t evaluated.
- Impact: Weakens any claim about market adjusted insight. Note as a data constraint and plan to fix.
3. No vectorized features in use:
- DTM/TF-IDF/curated files have 0 rows, so the classifier wasn’t actually driving the strategy.
- Impact: “sentiment driven” results are effectively baseline rules without ML signals; emphasize this as a limitation and a priority for future work.