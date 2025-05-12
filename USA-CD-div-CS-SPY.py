# Let's import the library
import yfinance as yf 
import pandas as pd
import numpy as np 
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
rGDP_data = np.log(
    fred.get_series('GDPC1',
                    observation_start='2019-05-11',
                    observation_end='2025-05-11')
)
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
xlp_data = xlp.history(period="5y")
xly_data = xly.history(period="5y")
spy_data = spy.history(period="5y")

# Extract the closing prices
xlp_close = (xlp_data['Close'])
xly_close = (xly_data['Close'])
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
ax2.plot(spy_close, label='SPY', color='g')
ax2.set_ylabel('SPY Price', color='g')
ax2.tick_params(axis='y', labelcolor='g')

# Title and legend
plt.title('XLY/XLP Ratio and SPY Over the Last 1 Year')
fig.tight_layout()
plt.show()

##################
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
