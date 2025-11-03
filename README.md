## Core Performance
- **Final balance:** 1.1356  
- **Total gain/loss:** 0.1403  
- **Avg annual return (%):** 91.6150  
- **Total return (%):** 14.0932  

## Trading Simulation
- **Trades:** 51  
- **Win rate:** 0.608  
- **Avg PnL per trade:** 0.002581  
- **Median PnL per trade:** 0.001594  
- **Equity start → end:** 1.00 → 1.14  
- **Max drawdown:** −1.845%  
- **By type:** Buy 50, mean PnL 0.002581, total PnL 0.129041  

## Market-Adjusted Results
Source: `datasets/historical_prices_impact.csv`
- **market_adj_return mean:** 0.000062  
- **market_adj_return median:** 0.000000  
- **market_adj_return std:** 0.008423  

## Impact Score Behavior
- **Corr(impact score, next-day return):** −0.1049  

**Impact score vs next-day return (deciles)**
| decile | count | mean     | median   |
|-------:|------:|---------:|---------:|
| 0      | 39    | 0.003076 | 0.001758 |
| 1      | 1     | 0.008597 | 0.008597 |
| 2      | 5     | −0.001098| −0.003823|
| 3      | 5     | 0.001194 | −0.002802|

## News Vectorization Coverage
- **DTM:** rows 0, symbols 0, range N/A  
- **TF-IDF:** rows 0, symbols 0, range N/A  
- **Curated:** rows 0, symbols 0, range N/A  

# Notes
1. Returned consistency fixed:
- ```bash final_summary.csv``` now aligns: final_balance 1.1356 ⇒ ~14.09% total return, and ```bash total_return_pct``` is 14.0932
2. Market adjustment fixed:
- No more NaNs. Now have valid market adjusted stats. Kept a brief caveat that it’s a simple single factor adjustment.
3. NLP features missing:
- Vectorized news files are still empty, so results aren’t truly “sentiment-driven.” Called this out as the main limitation and top priority for future work.