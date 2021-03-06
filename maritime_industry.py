!pip install yfinance
!pip install yahoofinancials

import pandas as pd
import yfinance as yf
from yahoofinancials import YahooFinancials
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
%matplotlib inline

# Evergreen
evergreen_df = yf.download('2603.TW', 
                      start='2021-01-01', 
                      end='2021-12-31', 
                      progress=False)

evergreen = evergreen_df['Close']
evergreen.describe()

# Yang Ming
yangming_df = yf.download('2609.TW', 
                      start='2021-01-01', 
                      end='2021-12-31', 
                      progress=False)

yangming = yangming_df['Close']
yangming.describe()

# Wan Hai
wanhai_df = yf.download('2615.TW', 
                      start='2021-01-01', 
                      end='2021-12-31', 
                      progress=False)

wanhai = wanhai_df['Close']
wanhai.describe()

# Combine the 3 dataframe
ship_df = pd.concat([evergreen.rename('Evergreen'), yangming.rename('Yang Ming'), wanhai.rename('Wan Hai')], axis=1)
ship_df

# TAIEX Index
tw_df = yf.download('^TWII', 
                      start='2021-01-01', 
                      end='2021-12-31', 
                      progress=False)

tw = tw_df['Close']
tw


ship_df.plot(subplots = True, title = 'Stock Data')
tw.plot(title = 'TAIEX')

# calculate daily stock_data returns
ship_returns = ship_df.pct_change()
ship_returns.plot()
ship_returns.describe()

# calculate daily benchmark_data returns
tw_returns = tw.pct_change()
tw_returns.plot()
tw_returns.describe()

# calculate the difference in daily returns
excess_returns = ship_returns.sub(tw_returns, axis = 0)
excess_returns.plot()
excess_returns.describe()

# calculate the mean of excess_returns 
avg_excess_return = excess_returns.mean()
avg_excess_return.plot.bar(title = 'Mean of the Return Difference')

# calculate the standard deviations
sd_excess_return = excess_returns.std()
sd_excess_return.plot.bar(title = 'Standard Deviation of the Return Difference')

# calculate the daily sharpe ratio
daily_sharpe_ratio = avg_excess_return.div(sd_excess_return)

# annualize the sharpe ratio
annual_factor = np.sqrt(252)
annual_sharpe_ratio = daily_sharpe_ratio.mul(annual_factor)

# plot the annualized sharpe ratio
annual_sharpe_ratio.plot.bar(title = 'Annualized Sharpe Ratio: Stocks vs TAIEX')
