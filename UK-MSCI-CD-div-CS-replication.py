import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import websockets
print(websockets.__version__)


#pip install yfinance[fast]
#pip uninstall yfinance

#########################
# Adding FTSE100 ##
####################
# FTSE 100 index ticker symbol
ftse_ticker = '^FTSE'

# MSCI UK Consumer Staples constituents https://www.msci.com/documents/10199/0c31f904-628b-4c2f-8c0d-b8bdcd87fd26 
staples_tickers = [
    'ULVR.L',  # Unilever PLC
    'BATS.L',  # British American Tobacco
    'DGE.L',   # Diageo
    'RKT.L',   # Reckitt Benckiser Group
    'HLN.L',   # Haleon
    'TSCO.L',  # Tesco
    'IMB.L',   # Imperial Brands
    'ABF.L',   # Associated British Foods
    'CCH.L',   # Coca-Cola HBC
    'SBRY.L'   # Sainsbury (J)
]

# MSCI UK Consumer Discretionary constituents https://www.msci.com/documents/10199/e03ad7c3-cd68-4a97-8712-3846390de7a0 
discretionary_tickers = [
    'CPG.L',  # Compass Group
    'IHG.L',  # InterContinental Hotels Group
    'NXT.L',  # Next
    'PSON.L', # Pearson
    'BTRW.L', # Barratt Developments
    'WTB.L',  # Whitbread
    'KGF.L',  # Kingfisher
    'TW.L',   # Taylor Wimpey
    'ENT.L',  # Entain
    'PSN.L'   # Persimmon
]
# Download FTSE 100 index data
#ftse_data = yf.download(ftse_ticker, start='2024-09-01', end='2025-05-30')['Close']
ftse_data = np.log(yf.download(ftse_ticker, start='2024-09-01', end='2025-05-30')['Close'])


# Download data for Consumer Staples
staples_data = yf.download(staples_tickers, start='2024-09-01', end='2025-05-30')['Close']

# Download data for Consumer Discretionary
discretionary_data = yf.download(discretionary_tickers, start='2024-09-01', end='2025-05-30')['Close']
# Drop rows with missing values
staples_data = staples_data.dropna()
discretionary_data = discretionary_data.dropna()
# Calculate weights for Consumer Staples
staples_weights = staples_data.div(staples_data.sum(axis=1), axis=0)

# Calculate weights for Consumer Discretionary
discretionary_weights = discretionary_data.div(discretionary_data.sum(axis=1), axis=0)
# Daily returns for Consumer Staples
staples_returns = staples_data.pct_change().dropna()

# Daily returns for Consumer Discretionary
discretionary_returns = discretionary_data.pct_change().dropna()
# Weighted returns for Consumer Staples
staples_weighted_returns = (staples_returns * staples_weights.shift(1)).sum(axis=1)
staples_index_value = (1 + staples_weighted_returns).cumprod()

# Weighted returns for Consumer Discretionary
discretionary_weighted_returns = (discretionary_returns * discretionary_weights.shift(1)).sum(axis=1)
discretionary_index_value = (1 + discretionary_weighted_returns).cumprod()
# Align indices
aligned_staples, aligned_discretionary = staples_index_value.align(discretionary_index_value, join='inner')

# Calculate Discretionary/Staples ratio
discretionary_staples_ratio = aligned_discretionary / aligned_staples

######################################################################################################################################################
# Plot ratio with FTSE 100 on secondary y-axis
fig, ax1 = plt.subplots(figsize=(12, 6))
# Plot Discretionary/Staples ratio
ax1.plot(discretionary_staples_ratio.index, discretionary_staples_ratio, color='purple', label='Discretionary/Staples Ratio')
ax1.set_xlabel('Date')
ax1.set_ylabel('Discretionary/Staples Ratio', color='purple')
ax1.tick_params(axis='y', labelcolor='purple')
ax1.grid(True)

# Create secondary y-axis for FTSE 100
ax2 = ax1.twinx()
ax2.plot(ftse_data.index, ftse_data, color='blue', label='FTSE 100 Index')
ax2.set_ylabel('FTSE 100 Index', color='blue')
ax2.tick_params(axis='y', labelcolor='blue')

# Add title and legends
plt.title('Consumer Discretionary/Consumer Staples Index Ratio and FTSE 100 Index')
fig.tight_layout()
fig.legend(loc='upper left', bbox_to_anchor=(0.1,0.9))

plt.show()
######################################################################################################################################################
# Turn both Discretionary/Staples ratio and FTSE 100 into percentage change since initial value
# and use a single y-axis for comparison

# Calculate percentage change since initial value for FTSE 100
#ftse_percentage_change = (ftse_data / ftse_data.iloc[0] - 1) * 100

# Calculate percentage change since initial value for Discretionary/Staples ratio
#discretionary_staples_percentage_change = (discretionary_staples_ratio / discretionary_staples_ratio.iloc[0] - 1) * 100

# Plot both series with a single y-axis
#fig, ax = plt.subplots(figsize=(12, 6))

# Plot percentage change for Discretionary/Staples ratio
#ax.plot(discretionary_staples_percentage_change.index, discretionary_staples_percentage_change, 
#        color='purple', label='Discretionary/Staples Ratio (% Change)')

# Plot percentage change for FTSE 100
#ax.plot(ftse_percentage_change.index, ftse_percentage_change, 
#        color='blue', label='FTSE 100 (% Change)')

# Add labels, title, and legend
#ax.set_xlabel('Date')
#ax.set_ylabel('Percentage Change (%)')
#ax.set_title('Percentage Change: Discretionary/Staples Ratio and FTSE 100 Index')
#ax.legend(loc='upper left')
#ax.grid(True)

#plt.tight_layout()
#plt.show()
