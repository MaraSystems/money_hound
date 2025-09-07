import streamlit as st
from datetime import datetime
import random
import sys
import os
import pandas as pd
import joblib
import numpy as np

from .auth import login
from . import dataframe
from lib import engineer, tracker, detector
from sklearn.preprocessing import RobustScaler, MinMaxScaler, LabelEncoder


BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "../../models", "predict_fraud_score_model_20250907_184121")
model = joblib.load(MODEL_PATH)

def predict(data):
    np.random.seed(42)
    unusual_data = get_unusual(data)

    st.subheader('Summary')
    data = unusual_data.drop(columns=['fraud', 'fraud_score'])
    pred = model.predict(data)

    classified = unusual_data.iloc[0]
    
    st.markdown(f"**Fraudulent:** {classified['fraud']}")
    st.markdown(f"**Fraud Score:** {classified['fraud_score']}")
    st.markdown(f"**Predicted Score:** {pred[0]}")


def get_unusual(data):
    data['date'] = str(data['date'])

    old_df = dataframe.engineered_transactions_df
    df = pd.concat([old_df, data]).set_index('time')

    discrete_features = df.select_dtypes(include=['object']).columns.tolist()
    encoded = engineer.encoder(df, discrete_features)
    df[discrete_features] = encoded

    scaler = RobustScaler()
    scaled_data = scaler.fit_transform(df)
    df = pd.DataFrame(scaled_data, columns=df.columns)

    df_unsual = df.copy()
    df_unsual = detector.unsual_amount(df_unsual)
    df_unsual = detector.unsual_balance(df_unsual)
    df_unsual = detector.unsual_location(df_unsual)
    df_unsual = detector.unsual_time(df_unsual)
    df_unsual = detector.unsual_device(df_unsual)

    df_fruad = engineer.anomalize(df_unsual, 'fraud')
    return df_fruad.tail(1)


def extract(data):
    transaction_limits = {1: 50_000, 2: 100_000, 3: 500_000, 4: 1_000_000}

    data['sub_account'] = data['holder_bvn'] == data['related_bvn']
    data['large_amount'] = transaction_limits[data['kyc']] < data['amount']
    data['balance_jump'] = -data['amount'] if data['type'] == 'DEBIT' else data['amount']
    data['previous_balance'] = data['balance'] - data['balance_jump']
    data['balance_jump_rate'] = data['balance_jump'] / max(data['previous_balance'], 1)
    data['balance_jump_rate_absolute'] = abs(data['balance_jump_rate'])
    data['drained_balance'] = data['balance_jump_rate'] < -.9
    data['pumped_balance'] = data['balance_jump_rate'] > .9
    data['large_amount_drain'] = data['large_amount'] & data['drained_balance']
    data['large_amount_pump'] = data['large_amount'] & data['pumped_balance']

    df = pd.DataFrame([data])
    df['hour'] = df['time'].dt.hour

    # Extracting the day
    df['week_day'] = df['time'].dt.day_name()

    # Extracting the month
    df['month'] = df['time'].dt.month_name()

    # Extracting the date
    df['date'] = df['time'].dt.date

    # Extracting the month day
    df['month_day'] = df['time'].dt.day.astype('object')
    
    time = df[['hour', 'week_day', 'month', 'date', 'month_day']].to_dict(orient='records')[0]
    data = { **data, **time }

    account = dataframe.get_account(data['holder'])
    holder_df = dataframe.get_transactions('holder', data['holder'])
    holder_bvn_df = dataframe.get_transactions('holder_bvn', data['holder_bvn'])
    related_df = dataframe.get_transactions('related', data['related'])
    related_bvn_df = dataframe.get_transactions('related_bvn', data['related_bvn'])

    data['distance_from_home (km)'] = engineer.distance_from_home(holder_df, data, 'holder')
    data['far_distance'] = data['distance_from_home (km)'] >= 100

    holder_count_frequency = {f'holder_{feature}_count_frequency': engineer.count_related(holder_df, data, target='holder', feature=feature) for feature in ['related', 'device', 'channel', 'state']}

    holder_bvn_count_frequency = {f'holder_bvn_{feature}_count_frequency': engineer.count_related(holder_bvn_df, data, target='holder_bvn', feature=feature) for feature in ['related_bvn', 'device', 'channel', 'state']}

    data = { **data, **holder_count_frequency, **holder_bvn_count_frequency }
    data['holder_device_has_history'] = data['holder_device_count_frequency'] > 0
    data['is_opening_device'] = data['device'] == account['opening_device']

    data_df = pd.DataFrame([data])
    columns = data_df.columns
    new_df = pd.concat([dataframe.engineered_transactions_df[columns].sample(5000), data_df], axis=0, ignore_index=True)

    new_df = engineer.get_bounds(new_df)
    new_df = engineer.get_holder_occurance(new_df)
    new_df = engineer.get_holder_bvn_occurance(new_df)
    new_df = engineer.get_related_occurance(new_df)
    new_df = engineer.get_related_bvn_occurance(new_df)
    
    new_df = tracker.get_rolling(new_df)
    return new_df.tail(1)


def atm_transaction():
    account = st.session_state['account']
    atm_list = dataframe.get_atms(account['latitude'], account['longitude'])
    
    with st.form('atm_transaction'):
        st.metric(label="💰 Balance", value=f"₦{account['balance']:,.2f}")
        st.subheader(f"🌍 State: {account['state']}")
        st.markdown(f"🆔 **KYC Level:** {account['kyc']}")

        transaction_amount = st.number_input("💵 Amount (₦)", min_value=100, step=100)

        transaction_atm = st.selectbox("🔢 ATM Stand", atm_list.index)

        transaction_category = st.selectbox("🔢 Category", ['WITHDRAWAL', 'PAYMENT', 'DEPOSIT'])

        submitted = st.form_submit_button("Transact")

    if submitted:
        atm = atm_list.loc[transaction_atm]
        location = {'state': atm['state'], 'latitude': atm['latitude'], 'longitude': atm['longitude']}
        tx_type =  'CREDIT' if transaction_category == 'DEPOSIT' else 'DEBIT'
        balance = account['balance'] + transaction_amount if tx_type == 'CREDIT' else account['balance'] - transaction_amount

        if balance < 0:
            st.write('Insufficient Fund')
            return
        
        data = { 
            'amount': transaction_amount,
            'balance': balance,
            'time': datetime.now(),
            'holder': account['account_no'],
            'holder_bvn': account['bvn'], 
            'related': atm['device_id'], 
            'related_bvn': atm['device_id'], 
            'related_bank': atm['bank_id'],
            **location,
            'status': 'SUCCESS',
            'type': tx_type,
            'category': transaction_category,
            'channel': 'CARD',
            'device': atm['device_id'],
            'nonce': str(random.randint(1e20, 1e21-1)),
            'reported': False,
            'kyc': int(account['kyc']),
            'merchant': True,
            'holder_latitude': account['latitude'],
            'holder_longitude': account['longitude']
        }

        extracted_data = extract(data)
        predict(extracted_data)


def pos_transaction():
    account = st.session_state['account']
    merchant_list = dataframe.get_merchants(account['latitude'], account['longitude'])
    
    with st.form('pos_transaction'):
        st.metric(label="💰 Balance", value=f"₦{account['balance']:,.2f}")
        st.subheader(f"🌍 State: {account['state']}")
        st.markdown(f"🆔 **KYC Level:** {account['kyc']}")

        transaction_amount = st.number_input("💵 Amount (₦)", min_value=100, step=100, max_value=int(account['balance']))

        transaction_merchant = st.selectbox("🔢 Merchant", merchant_list.index)

        transaction_category = st.selectbox("🔢 Category", ['WITHDRAWAL', 'PAYMENT', 'BILL'])

        submitted = st.form_submit_button("Transact")

    if submitted:
        merchant = merchant_list.loc[transaction_merchant]
        device = random.choice(merchant['devices'])
        location = {'state': merchant['state'], 'latitude': merchant['latitude'], 'longitude': merchant['longitude']}

        data = { 
            'amount': transaction_amount,
            'balance': account['balance'] - transaction_amount,
            'time': datetime.now(),
            'holder': account['account_no'],
            'holder_bvn': account['bvn'], 
            'related': merchant['account_no'], 
            'related_bvn': merchant['bvn'], 
            'related_bank': merchant['bank_id'],
            **location,
            'status': 'SUCCESS',
            'type': 'DEBIT',
            'category': transaction_category,
            'channel': 'CARD',
            'device': device,
            'nonce': str(random.randint(1e20, 1e21-1)),
            'reported': False,
            'kyc': int(account['kyc']),
            'merchant': True,
            'holder_latitude': account['latitude'],
            'holder_longitude': account['longitude']
        }

        extracted_data = extract(data)
        predict(extracted_data)


def mobile_transaction():
    account = st.session_state['account']
    states = dataframe.list_states()
    state_index = states.tolist().index(account['state'])

    with st.form('mobile_transaction'):
        st.metric(label="💰 Balance", value=f"₦{account['balance']:,.2f}")
        st.subheader(f"🌍 State: {account['state']}")
        st.markdown(f"🆔 **KYC Level:** {account['kyc']}")

        transaction_amount = st.number_input("💵 Amount (₦)", min_value=100, step=100, max_value=int(account['balance']))

        transaction_destination = st.selectbox("🔢 Destination Account", dataframe.list_accounts(account['account_no']))

        transaction_state = st.selectbox("📍 Transaction State", states, index=state_index)

        transaction_channel = st.selectbox("🔢 Channel", ['APP', 'USSD'])

        random_device = st.checkbox("🔢 Random Device")

        submitted = st.form_submit_button("Transact")

    if submitted:
        related_account = dataframe.get_account(transaction_destination)
        location = dataframe.select_location(transaction_state)
        device = random.choice(account['devices']) if not random_device else random.choice(dataframe.list_devices())

        data = { 
            'amount': transaction_amount,
            'balance': account['balance'] - transaction_amount,
            'time': datetime.now(),
            'holder': account['account_no'],
            'holder_bvn': account['bvn'], 
            'related': transaction_destination, 
            'related_bvn': related_account['bvn'], 
            'related_bank': related_account['bank_id'],
            **location,
            'status': 'SUCCESS',
            'type': 'DEBIT',
            'category': 'TRANSFER',
            'channel': transaction_channel,
            'device': device,
            'nonce': str(random.randint(1e20, 1e21-1)),
            'reported': False,
            'kyc': account['kyc'],
            'merchant': account['merchant'],
            'holder_latitude': account['latitude'],
            'holder_longitude': account['longitude']
        }

        extracted_data = extract(data)
        predict(extracted_data)
    

EVENTS = {
    'MOBILE TRANSACTION': mobile_transaction,
    'POS TRANSACTION': pos_transaction,
    'ATM TRANSACTION': atm_transaction
}


def detect():
    event_keys = list(EVENTS.keys())
    tabs = st.tabs(event_keys)
    
    for i, tab in enumerate(tabs):
        with tab:
            EVENTS[event_keys[i]]()
   

def main():
    st.title("Fraud Detector")
    
    if "account" not in st.session_state:
        login('DETECT')
    else:
        detect()



