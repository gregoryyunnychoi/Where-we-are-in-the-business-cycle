# Let's import the library
import yfinance as yf 

# Pandas is used for dataframe and tabular data manipulation
import pandas as pd 
import matplotlib.pyplot as plt
from fredapi import Fred
import simfin as sf
import os

# Create Ticker objects for XLP and XLY
xlp = yf.Ticker("XLP")
xly = yf.Ticker("XLY")

# Download historical data for the past 5 years
xlp_data = xlp.history(period="5y")
xly_data = xly.history(period="5y")

# Extract the closing prices
xlp_close = xlp_data['Close']
xly_close = xly_data['Close']

# Calculate XLY/XLP ratio
xly_xlp_ratio = xly_close / xlp_close

# Plot the XLY/XLP ratio
plt.figure(figsize=(10, 6))
plt.plot(xly_xlp_ratio, label='XLY/XLP Ratio', color='b')
plt.xlabel('Date')
plt.ylabel('Ratio')
plt.title('XLY/XLP Ratio Over the Last 5 Years')
plt.legend()
plt.grid(True)
plt.show()
####
## Adding GDP ###

# Create a Fred object with your API key (you'll need to get your own key from https://fred.stlouisfed.org/)
fred = Fred(api_key='42af668b5078244d37f40c133816843a')
# Download US real GDP data from FRED (quarterly data)
rGDP_data = fred.get_series('GDPC1', observation_start='2019-01-01', observation_end='2024-11-01')

# Resample rGDP to daily to match with XLY/XLP ratio
rGDP_daily = rGDP_data.resample('D').ffill()

# Plot XLY/XLP ratio along with US rGDP using secondary y-axis for GDP
fig, ax1 = plt.subplots(figsize=(12, 8))

# Plot XLY/XLP ratio on the primary y-axis
ax1.plot(xly_xlp_ratio, label='XLY/XLP Ratio', color='b')
ax1.set_xlabel('Date')
ax1.set_ylabel('XLY/XLP Ratio', color='b')
ax1.tick_params(axis='y', labelcolor='b')
ax1.grid(True)

# Plot rGDP on the secondary y-axis
ax2 = ax1.twinx()
ax2.plot(rGDP_daily, label='US Real GDP', color='r', linestyle='--')
ax2.set_ylabel('US Real GDP (Billions of Chained 2012 Dollars)', color='r')
ax2.tick_params(axis='y', labelcolor='r')

# Title and legend
plt.title('XLY/XLP Ratio and US Real GDP Over the Last 5 Years')
fig.tight_layout()
plt.show()

####Now replace GDP with SPY. 
# Create Ticker objects for XLP, XLY, and SPY
xlp = yf.Ticker("XLP")
xly = yf.Ticker("XLY")
spy = yf.Ticker("SPY")

# Download historical data for the past 5 years
xlp_data = xlp.history(period="10y")
xly_data = xly.history(period="10y")
spy_data = spy.history(period="10y")

# Extract the closing prices
xlp_close = xlp_data['Close']
xly_close = xly_data['Close']
spy_close = spy_data['Close']

# Calculate XLY/XLP ratio
xly_xlp_ratio = xly_close / xlp_close

# Plot XLY/XLP ratio along with SPY using secondary y-axis for SPY
fig, ax1 = plt.subplots(figsize=(12, 8))

# Plot XLY/XLP ratio on the primary y-axis
ax1.plot(xly_xlp_ratio, label='XLY/XLP Ratio', color='b')
ax1.set_xlabel('Date')
ax1.set_ylabel('XLY/XLP Ratio', color='b')
ax1.tick_params(axis='y', labelcolor='b')
ax1.grid(True)

# Plot SPY on the secondary y-axis
ax2 = ax1.twinx()
ax2.plot(spy_close, label='SPY', color='g', linestyle='--')
ax2.set_ylabel('SPY Price', color='g')
ax2.tick_params(axis='y', labelcolor='g')

# Title and legend
plt.title('XLY/XLP Ratio and SPY Over the Last 5 Years')
fig.tight_layout()
plt.show()


######################################
# Europe ##
import yfinance as yf
import matplotlib.pyplot as plt

# Define the tickers for consumer discretionary and consumer staples
discretionary_ticker = 'SPYR.DE'  # Lyxor STOXX Europe 600 Consumer Discretionary ETF
staples_ticker = 'SPYC.DE'  # iShares STOXX Europe 600 Consumer Staples ETF
stoxx_ticker = '^STOXX'  # STOXX Europe 600 Index

# Fetch data
discretionary_etf = yf.Ticker(discretionary_ticker)
staples_etf = yf.Ticker(staples_ticker)
stoxx_index = yf.Ticker(stoxx_ticker)

# Get historical market data for the past 5 years
discretionary_hist = discretionary_etf.history(period="5y")
staples_hist = staples_etf.history(period="5y")
stoxx_hist = stoxx_index.history(period="5y")

# Extract the closing prices
discretionary_close = discretionary_hist['Close']
staples_close = staples_hist['Close']
stoxx_close = stoxx_hist['Close']

# Calculate Consumer Discretionary / Consumer Staples Ratio
ratio = discretionary_close / staples_close

# Plot Consumer Discretionary / Consumer Staples ratio along with STOXX on secondary y-axis
fig, ax1 = plt.subplots(figsize=(12, 8))

# Plot Consumer Discretionary/Staples ratio on the primary y-axis
ax1.plot(ratio.index, ratio, label='Consumer Discretionary / Consumer Staples Ratio', color='b')
ax1.set_xlabel('Date')
ax1.set_ylabel('Consumer Discretionary / Consumer Staples Ratio', color='b')
ax1.tick_params(axis='y', labelcolor='b')
ax1.grid(True)

# Plot STOXX on the secondary y-axis
ax2 = ax1.twinx()
ax2.plot(stoxx_close.index, stoxx_close, label='STOXX Europe 600', color='r', linestyle='--')
ax2.set_ylabel('STOXX Europe 600', color='r')
ax2.tick_params(axis='y', labelcolor='r')

# Title and legend
plt.title('Consumer Discretionary / Consumer Staples Ratio and STOXX Europe 600 Over the Last 5 Years')
fig.tight_layout()
plt.show()

###################
## New Order Index ###
###################

import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from fredapi import Fred
import numpy as np

# Initialize FRED with your API key
fred = Fred(api_key='42af668b5078244d37f40c133816843a')

# Fetch the Manufacturers' New Orders data from FRED (series ID: NEWORDER)
new_orders_data = fred.get_series('NEWORDER', observation_start='2015-01-01')

# Convert to a DataFrame and set proper datetime index
new_orders_data = pd.Series(new_orders_data)
new_orders_data.index = pd.to_datetime(new_orders_data.index)

# Resample new orders data to daily to match SPY for visualization purposes
new_orders_daily = np.log(new_orders_data.resample('D').ffill())
# Fetch SPY data using yfinance for the last 5 years
spy = yf.Ticker("SPY")
spy_data = np.log(spy.history(period="10y"))

# Extract the closing prices for SPY
spy_close = spy_data['Close']

# Plot New Orders index along with SPY using secondary y-axis for SPY
fig, ax1 = plt.subplots(figsize=(12, 8))

# Plot New Orders on the primary y-axis
ax1.plot(new_orders_daily.index, new_orders_daily, label='U.S. New Orders: Nondefense Capital Goods Excluding Aircraft', color='b')
ax1.set_xlabel('Date')
ax1.set_ylabel('New Orders (in Millions of Dollars)', color='b')
ax1.tick_params(axis='y', labelcolor='b')
ax1.grid(True)

# Plot SPY on the secondary y-axis
ax2 = ax1.twinx()
ax2.plot(spy_close.index, spy_close, label='SPY', color='g', linestyle='--')
ax2.set_ylabel('SPY Price', color='g')
ax2.tick_params(axis='y', labelcolor='g')

# Title and legend
plt.title("U.S. New Orders (Nondefense Capital Goods Excluding Aircraft) and SPY Over the Last 5 Years")
fig.tight_layout()
plt.show()


### Sim Fin API###

sf.set_api_key('5e2d0911-53fe-4f5e-957d-d474fbbc5330')
# Setup a place (local directory) to store the data 
script_directory = os.path.dirname(os.path.abspath(__file__))
sf.set_data_dir('script_directory')
#