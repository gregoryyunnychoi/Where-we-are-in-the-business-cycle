######################################
# Europe ##
# Let's import the library
import yfinance as yf 
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from fredapi import Fred
import simfin as sf
import os

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
