import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# List of MSCI UK Consumer Staples constituents tickers
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

# Set the date range
start_date = "2024-01-01"
end_date = "2025-01-05"

# Initialize a dictionary to store data
staples_data = {}

for ticker in staples_tickers:
    # Download historical data using yfinance
    stock_data = yf.download(ticker, start=start_date, end=end_date)

    # Retrieve stock information for shares outstanding
    stock_info = yf.Ticker(ticker).info
    shares_outstanding = stock_info.get("sharesOutstanding", None)

    # Check if shares outstanding data is available
    if shares_outstanding:
        # Calculate Market Cap = Shares Outstanding * Close Price
        stock_data['Market_Cap'] = shares_outstanding * stock_data['Close']
        staples_data[ticker] = stock_data[['Market_Cap']]
    else:
        print(f"Shares outstanding data not available for {ticker}.")

# Combine all market cap data into a single DataFrame
staples_cap_df = pd.concat(staples_data, axis=1)
staples_cap_df.columns = [f"{ticker}_Market_Cap" for ticker in staples_data.keys()]

# Handle missing data by forward-filling and dropping remaining NaN
staples_cap_df = staples_cap_df.fillna(method='ffill').dropna()

# Calculate weights for Consumer Staples
staples_weights = staples_cap_df.div(staples_cap_df.sum(axis=1).replace(0, 1), axis=0)

# Daily returns for Consumer Staples
staples_returns = staples_cap_df.pct_change().dropna()

# Align weights and returns
staples_weights = staples_weights.loc[staples_returns.index]

# Weighted returns for Consumer Staples
staples_weighted_returns = (staples_returns * staples_weights.shift(1)).sum(axis=1)

# Cumulative index value for Consumer Staples
staples_index_value = (1 + staples_weighted_returns).cumprod()

####### ##########
# Discretionary ##
#################
# List of MSCI UK Consumer Discretionary constituents tickers
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

# Initialize a dictionary to store data
discretionary_data = {}

for ticker in discretionary_tickers:
    # Download historical data using yfinance
    stock_data = yf.download(ticker, start=start_date, end=end_date)

    # Retrieve stock information for shares outstanding
    stock_info = yf.Ticker(ticker).info
    shares_outstanding = stock_info.get("sharesOutstanding", None)

    # Check if shares outstanding data is available
    if shares_outstanding:
        # Calculate Market Cap = Shares Outstanding * Close Price
        stock_data['Market_Cap'] = shares_outstanding * stock_data['Close']
        discretionary_data[ticker] = stock_data[['Market_Cap']]
    else:
        print(f"Shares outstanding data not available for {ticker}.")

# Combine all market cap data into a single DataFrame
discretionary_cap_df = pd.concat(discretionary_data, axis=1)
discretionary_cap_df.columns = [
    f"{ticker}_Market_Cap" for ticker in discretionary_tickers if ticker in discretionary_data
]

# Calculate weights for Consumer Discretionary
discretionary_weights = discretionary_cap_df.div(discretionary_cap_df.sum(axis=1), axis=0)

# Daily returns for Consumer Discretionary
discretionary_returns = discretionary_cap_df.pct_change().dropna()

# Weighted returns for Consumer Discretionary
discretionary_weighted_returns = (discretionary_returns * discretionary_weights.shift(1)).sum(axis=1)

# Cumulative index value for Consumer Discretionary
discretionary_index_value = (1 + discretionary_weighted_returns).cumprod()


###########
##Aligning#
###########

# Align the two indices (staples_index_value and discretionary_index_value) based on their common dates
aligned_index = staples_index_value.index.intersection(discretionary_index_value.index)

# Subset both indices to match the common dates
aligned_staples = staples_index_value.loc[aligned_index]
aligned_discretionary = discretionary_index_value.loc[aligned_index]

# Calculate the ratio of discretionary to staples
discretionary_to_staples_ratio = aligned_discretionary / aligned_staples

###############
#Visualisation#
###############

# Plot the Discretionary/Staples Ratio
plt.figure(figsize=(12, 6))
plt.plot(discretionary_to_staples_ratio, label="Discretionary/Staples Ratio", color="blue")
plt.title("Discretionary/Staples Ratio Over Time")
plt.xlabel("Date")
plt.ylabel("Ratio")
plt.legend()
plt.grid(True)
plt.show()

##################
#Adding FTSE data#
##################

# FTSE 100 index ticker symbol
ftse_ticker = '^FTSE'
 
# Fetch FTSE index data
ftse_data = yf.download(ftse_ticker, start=start_date, end=end_date)["Close"]

# Align FTSE data with discretionary_to_staples_ratio
aligned_ftse = ftse_data.loc[discretionary_to_staples_ratio.index]

# Plot the Discretionary/Staples Ratio with FTSE index on the secondary y-axis
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot Discretionary/Staples Ratio on the left y-axis
ax1.plot(discretionary_to_staples_ratio, label="Discretionary/Staples Ratio", color="blue")
ax1.set_xlabel("Date")
ax1.set_ylabel("Discretionary/Staples Ratio", color="blue")
ax1.tick_params(axis="y", labelcolor="blue")
ax1.grid(True)

# Create a secondary y-axis for FTSE index
ax2 = ax1.twinx()
ax2.plot(aligned_ftse, label="FTSE Index", color="red", linestyle="--")
ax2.set_ylabel("FTSE Index", color="red")
ax2.tick_params(axis="y", labelcolor="red")

# Add a title and legend
plt.title("Discretionary/Staples Ratio and FTSE Index Over Time")
fig.tight_layout()
plt.legend(loc="upper left")
plt.show()