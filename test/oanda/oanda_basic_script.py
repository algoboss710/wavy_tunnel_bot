import oandapyV20
import oandapyV20.endpoints.accounts as accounts
import requests
from datetime import datetime
import pytz

API_TOKEN = ""
ACCOUNT_ID = ""
API_URL = "https://api-fxtrade.oanda.com/v3"

# Initialize the OANDA API client
client = oandapyV20.API(access_token=API_TOKEN)

# Function to get account information
def get_account_info():
    r = accounts.AccountSummary(accountID=ACCOUNT_ID)
    response = client.request(r)
    return response

# Function to check if the market is open
def is_market_open():
    url = f"{API_URL}/instruments/EUR_USD/candles?count=1&granularity=S5"
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    response = requests.get(url, headers=headers).json()
    time = response['candles'][0]['time']
    return time

# Function to get current date and time in your timezone
def get_current_time():
    local_tz = pytz.timezone('America/New_York')  # Adjust to your time zone
    now = datetime.now(local_tz)
    return now.strftime("%Y-%m-%d %H:%M:%S %Z")

# Get account info
account_info = get_account_info()
print("Account Info:", account_info)

# Check if market is open
market_time = is_market_open()
print("Last Market Data Time:", market_time)

# Get current time in your timezone
current_time = get_current_time()
print("Current Time in Your Timezone:", current_time)
