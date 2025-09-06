import os
import pandas as pd

from lib import tracker

BASE_DIR = os.path.dirname(__file__)  
DATA_DIR = "../../datasets"
TRANSACTIONS_PATH = os.path.join(BASE_DIR, DATA_DIR, "transactions.csv")
BANK_DEVICES_PATH = os.path.join(BASE_DIR, DATA_DIR, "bank_devices.csv")
ACCOUNTS_PATH = os.path.join(BASE_DIR, DATA_DIR, "accounts.csv")
CLASSIFIED_TRANSACTIONS = os.path.join(BASE_DIR, DATA_DIR, "classified_transactions.csv")
ANALYZED_TRANSACTIONS = os.path.join(BASE_DIR, DATA_DIR, "analyzed_transactions.csv")
ENGINEERED_TRANSACTIONS = os.path.join(BASE_DIR, DATA_DIR, "engineered_transactions.csv")
LOCATION_PATH = os.path.join(BASE_DIR, DATA_DIR, "nigerian_state_locations.csv")


location_df = pd.read_csv(LOCATION_PATH)
accounts_df = pd.read_csv(ACCOUNTS_PATH)
analyzed_transactions_df = pd.read_csv(ANALYZED_TRANSACTIONS, parse_dates=['time'])
engineered_transactions_df = pd.read_csv(ENGINEERED_TRANSACTIONS, parse_dates=['time'])
classified_transactions_df = pd.read_csv(CLASSIFIED_TRANSACTIONS)
bank_devices_df = pd.read_csv(BANK_DEVICES_PATH)
transactions_df = pd.read_csv(TRANSACTIONS_PATH)


accounts_df['devices'] = accounts_df['devices'].apply(lambda x: x.replace('[', '').replace(']', '').replace('\'', '').replace(' ', '').split(','))


def list_accounts(exclude=''):
    accounts = [item for item in analyzed_transactions_df['holder'].unique() if item not in exclude]
    return accounts


def get_account(account_no):
    return accounts_df[accounts_df['account_no'] == account_no].to_dict(orient='records')[0]


def list_states():
    return location_df['state'].unique()


def select_location(state):
    return location_df[location_df['state'] == state].sample(1).to_dict(orient='records')[0]


def list_devices():
    return accounts_df.sample(n=1).iloc[0]['devices']


def get_transactions(target, value):
    return engineered_transactions_df[engineered_transactions_df[target] == value]


def get_merchants(lat, lon):
    merchant_list = accounts_df[accounts_df['merchant']].copy()
    merchant_list['distance'] = merchant_list.apply(
        lambda merchant: 
        tracker.distance(
            latA=lat,
            lonA=lon,
            latB=merchant['latitude'],
            lonB=merchant['longitude']
        ),
        axis=1
    )
    merchant_list.sort_values(by='distance', inplace=True)
    merchant_list.index = merchant_list['account_no'] + ' ' + merchant_list['state'] + ' ('+ merchant_list['distance'].astype(str).str.slice(stop=4) + 'KM)'
    return merchant_list


def get_atms(lat, lon):
    atm_list = bank_devices_df.copy()
    atm_list['distance'] = atm_list.apply(
        lambda merchant: 
        tracker.distance(
            latA=lat,
            lonA=lon,
            latB=merchant['latitude'],
            lonB=merchant['longitude']
        ),
        axis=1
    )
    atm_list.sort_values(by='distance', inplace=True)
    atm_list.index = atm_list['device_id'] + ' ' + atm_list['state'] + ' ('+ atm_list['distance'].astype(str).str.slice(stop=4) + 'KM)'
    return atm_list