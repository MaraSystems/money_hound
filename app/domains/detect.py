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
from lib import engineer, tracker
from sklearn.preprocessing import RobustScaler, LabelEncoder


BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "../../models", "predict_fraud_score_model_20250906_073426")
model = joblib.load(MODEL_PATH)

def predict(data):
    np.random.seed(42)
    unusual_data = get_unusual(data)
    labels = unusual_data.iloc[0][['fraud', 'fraud_score']]

    st.subheader('Isolation Forest Classification')
    st.write(labels)

    st.subheader('XGBoost Prediction')
    data = unusual_data.drop(columns=['fraud', 'fraud_score'])
    pred = model.predict(data)
    st.write(pred)


def get_unusual(data):
    data['geo'] = str(data['geo'])
    data['date'] = str(data['date'])

    new_df = pd.DataFrame([data])
    old_df = dataframe.engineered_transactions_df
    df = pd.concat([old_df, new_df]).set_index('time')

    discrete_features = df.select_dtypes(include=['object', 'bool']).columns.tolist()
    encoded = engineer.encoder(df, discrete_features)
    df[discrete_features] = encoded

    scaler = RobustScaler()
    scaled_data = scaler.fit_transform(df)
    df = pd.DataFrame(scaled_data, columns=df.columns)

    df_unsual = df.copy()

    unsual_amount_columns = ['holder', 'large_amount_drain', 'large_amount_pump', 'holder_amount_bound_frequency', 'holder_large_amount_drain_True_occurance', 'holder_large_amount_pump_True_occurance', 'holder_amount_avg_1D', 'holder_amount_avg_7D', 'holder_amount_avg_30D', 'holder_amount_avg_120D']
    df_unsual_amount = engineer.anomalize(df[unsual_amount_columns], 'unsual_amount')
    df_unsual[['unsual_amount_score', 'unsual_amount']] = df_unsual_amount[['unsual_amount_score', 'unsual_amount']]

    unsual_balance_columns = ['holder', 'balance_jump', 'balance_jump_rate',
    'drained_balance', 'pumped_balance', 'holder_balance_jump_bound_frequency', 'holder_balance_jump_rate_bound_frequency', 'holder_drained_balance_True_occurance', 'holder_pumped_balance_True_occurance', 'holder_balance_avg_1D', 'holder_balance_avg_7D', 'holder_balance_avg_30D', 'holder_balance_avg_120D']
    df_unsual_balance = engineer.anomalize(df[unsual_balance_columns], 'unsual_balance')
    df_unsual[['unsual_balance_score', 'unsual_balance']] = df_unsual_balance[['unsual_balance_score', 'unsual_balance']]

    unsual_location_columns = ['holder', 'holder_state_count_frequency', 'distance_from_last (km)', 'holder_distance_from_last (km)_avg_1D', 'holder_distance_from_last (km)_avg_7D', 'holder_distance_from_last (km)_avg_30D', 'holder_distance_from_last (km)_avg_120D']
    df_unsual_location = engineer.anomalize(df[unsual_location_columns], 'unsual_location')
    df_unsual[['unsual_location_score', 'unsual_location']] = df_unsual_location[['unsual_location_score', 'unsual_location']]

    unsual_time_columns = ['holder', 'hour', 'holder_hour_bound_frequency', 'duration_from_last (hr)', 'holder_duration_from_last (hr)_avg_1D', 'holder_duration_from_last (hr)_avg_7D', 'holder_duration_from_last (hr)_avg_30D', 'holder_duration_from_last (hr)_avg_120D']
    df_unsual_time = engineer.anomalize(df[unsual_time_columns], 'unsual_time')
    df_unsual[['unsual_time_score', 'unsual_time']] = df_unsual_time[['unsual_time_score', 'unsual_time']]

    unsual_speed_columns = ['holder', 'speed_from_last (km/hr)', 'holder_speed_from_last (km/hr)_avg_1D', 'holder_speed_from_last (km/hr)_avg_7D', 'holder_speed_from_last (km/hr)_avg_30D', 'holder_speed_from_last (km/hr)_avg_120D']
    df_unsual_speed = engineer.anomalize(df[unsual_speed_columns], 'unsual_speed')
    df_unsual[['unsual_speed_score', 'unsual_speed']] = df_unsual_speed[['unsual_speed_score', 'unsual_speed']]

    unsual_device_columns = ['holder', 'holder_device_count_frequency', 'holder_holder_device_count_frequency_avg_1D', 'holder_holder_device_count_frequency_avg_7D', 'holder_holder_device_count_frequency_avg_30D', 'holder_holder_device_count_frequency_avg_120D']
    df_unsual_device = engineer.anomalize(df[unsual_device_columns], 'unsual_device')
    df_unsual[['unsual_device_score', 'unsual_device']] = df_unsual_device[['unsual_device_score', 'unsual_device']]

    unsual_reported_columns = ['holder', 'holder_reported_True_occurance', 'holder_holder_reported_True_occurance_avg_1D', 'holder_holder_reported_True_occurance_avg_7D', 'holder_holder_reported_True_occurance_avg_30D', 'holder_holder_reported_True_occurance_avg_120D']
    df_unsual_reported = engineer.anomalize(df[unsual_reported_columns], 'unsual_reported')
    df_unsual[['unsual_reported_score', 'unsual_reported']] = df_unsual_reported[['unsual_reported_score', 'unsual_reported']]
    df_unsual[['unsual_reported_score', 'unsual_reported']].corr()

    unsual_reversal_columns = ['holder', 'holder_category_REVERSAL_occurance', 'holder_holder_category_REVERSAL_occurance_avg_1D', 'holder_holder_category_REVERSAL_occurance_avg_7D', 'holder_holder_category_REVERSAL_occurance_avg_30D', 'holder_holder_category_REVERSAL_occurance_avg_120D']
    df_unsual_reversal = engineer.anomalize(df[unsual_reversal_columns], 'unsual_reversal')
    df_unsual[['unsual_reversal_score', 'unsual_reversal']] = df_unsual_reversal[['unsual_reversal_score', 'unsual_reversal']]

    unsual_related_columns = ['holder', 'holder_related_count_frequency', 'holder_holder_related_count_frequency_avg_1D', 'holder_holder_related_count_frequency_avg_7D', 'holder_holder_related_count_frequency_avg_30D', 'holder_holder_related_count_frequency_avg_120D']
    df_unsual_related = engineer.anomalize(df[unsual_related_columns], 'unsual_related')
    df_unsual[['unsual_related_score', 'unsual_related']] = df_unsual_related[['unsual_related_score', 'unsual_related']]

    unsual_related_bvn_columns = ['holder_bvn', 'holder_bvn_related_bvn_count_frequency', 'holder_holder_bvn_related_bvn_count_frequency_avg_1D', 'holder_holder_bvn_related_bvn_count_frequency_avg_7D', 'holder_holder_bvn_related_bvn_count_frequency_avg_30D', 'holder_holder_bvn_related_bvn_count_frequency_avg_120D']
    df_unsual_related_bvn = engineer.anomalize(df[unsual_related_bvn_columns], 'unsual_related_bvn')
    df_unsual[['unsual_related_bvn_score', 'unsual_related_bvn']] = df_unsual_related_bvn[['unsual_related_bvn_score', 'unsual_related_bvn']]

    df_fruad = engineer.anomalize(df_unsual, 'fraud')
    return df_fruad.tail(1)


def extract(data):
    data['sub_account'] = data['holder_bvn'] == data['related_bvn']
    data['large_amount'] = data['amount'] >= 100000
    data['balance_jump'] = -data['amount'] if data['type'] == 'DEBIT' else data['amount']
    data['previous_balance'] = data['balance'] - data['balance_jump']
    data['balance_jump_rate'] = data['balance_jump'] / max(data['previous_balance'], 1)
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


    holder_df = dataframe.get_transactions('holder', data['holder'])
    holder_bvn_df = dataframe.get_transactions('holder_bvn', data['holder_bvn'])
    related_df = dataframe.get_transactions('related', data['related'])
    related_bvn_df = dataframe.get_transactions('related_bvn', data['related_bvn'])

    data['geo'] = (data['latitude'], data['longitude'])
    data['distance_from_last (km)'] = engineer.distance_from_last(holder_df, data, 'holder')
    data['duration_from_last (hr)'] = engineer.duration_from_last(holder_df, data, 'holder')
    data['speed_from_last (km/hr)'] = data['distance_from_last (km)'] / max(data['duration_from_last (hr)'], 1)

    holder_count_frequency = {f'holder_{feature}_count_frequency': engineer.count_related(holder_df, data, target='holder', feature=feature) for feature in ['related', 'device', 'channel', 'state']}

    holder_bvn_count_frequency = {f'holder_bvn_{feature}_count_frequency': engineer.count_related(holder_bvn_df, data, target='holder_bvn', feature=feature) for feature in ['related_bvn', 'device', 'channel', 'state']}

    data = { **data, **holder_count_frequency, **holder_bvn_count_frequency }

    bound_relatives = [
        # Has user ever transacted around this hour
        { 'name': 'hour', 'bound': lambda x: (x-1, x+1) }, 

        # Has user ever had balance around this balance
        { 'name': 'balance', 'bound': lambda x: (x*.5, x*1.5) }, 

        # Has user ever made a transaction around this amount
        { 'name': 'amount', 'bound': lambda x: (x*.5, x*1.5) },
        
        # Has user balance ever jumped like this before
        { 'name': 'balance_jump', 'bound': lambda x: (x * 0.5, x * 1.5) },

        # Relative balance jump rate (percentage-like scaling)
        { 'name': 'balance_jump_rate', 'bound': lambda x: (x - 0.2, x + 0.2) }
    ]
    
    holder_bound_relatives = {f"holder_{feature['name']}_bound_frequency": engineer.bound_relation(holder_df, data, target='holder', feature=feature) for feature in bound_relatives}

    holder_bvn_bound_relatives = {f"holder_bvn_{feature['name']}_bound_frequency": engineer.bound_relation(holder_bvn_df, data, target='holder_bvn', feature=feature) for feature in bound_relatives}

    data = { **data, **holder_bound_relatives, **holder_bvn_bound_relatives }

    
    occurance_relatives = [
        { 'name': 'reported', 'value': True },
        { 'name': 'category', 'value': 'REVERSAL' },
        { 'name': 'drained_balance', 'value': True },
        { 'name': 'pumped_balance', 'value': True },
        { 'name': 'large_amount_drain', 'value': True },
        { 'name': 'large_amount_pump', 'value': True },
    ]

    holder_occurrence_count = {f"holder_{feature['name']}_{feature['value']}_occurance": engineer.count_occurrence(holder_df, data, target='holder', feature=feature['name'], value=feature['value']) for feature in occurance_relatives}

    holder_bvn_occurrence_count = {f"holder_bvn_{feature['name']}_{feature['value']}_occurance": engineer.count_occurrence(holder_bvn_df, data, target='holder_bvn', feature=feature['name'], value=feature['value']) for feature in occurance_relatives}

    related_occurrence_count = {f"related_{feature['name']}_{feature['value']}_occurance": engineer.count_occurrence(related_df, data, target='related', feature=feature['name'], value=feature['value']) for feature in occurance_relatives}

    related_bvn_occurrence_count = {f"related_bvn_{feature['name']}_{feature['value']}_occurance": engineer.count_occurrence(related_bvn_df, data, target='related_bvn', feature=feature['name'], value=feature['value']) for feature in occurance_relatives}

    data = { **data, **holder_occurrence_count, **holder_bvn_occurrence_count, **related_occurrence_count, **related_bvn_occurrence_count }

    data_df = pd.DataFrame([data])
    new_holder_df = pd.concat([holder_df, data_df])
    new_holder_bvn_df = pd.concat([holder_bvn_df, data_df])

    rolling_features = [
        # Transaction dynamics
        'amount', 'balance', 'balance_jump', 'balance_jump_rate',

        # Distance and Time
        'distance_from_last (km)', 'duration_from_last (hr)', 'speed_from_last (km/hr)', 
        
        # Device usage
        'holder_device_count_frequency', 

        # Holder - Related relationship
        'holder_related_count_frequency', 'holder_bvn_related_bvn_count_frequency',
        
        # Reversal Tracking
        'holder_category_REVERSAL_occurance', 'holder_bvn_category_REVERSAL_occurance',

        # Reported Tracking
        'holder_reported_True_occurance', 'holder_bvn_reported_True_occurance'
    ]

    holder_window_1 = tracker.rolling_averages(new_holder_df, 'holder', rolling_features, 1).iloc[-1]
    holder_bvn_window_1 = tracker.rolling_averages(new_holder_bvn_df, 'holder_bvn', rolling_features, 1).iloc[-1]

    holder_window_7 = tracker.rolling_averages(new_holder_df, 'holder', rolling_features, 7).iloc[-1]
    holder_bvn_window_7 = tracker.rolling_averages(new_holder_bvn_df, 'holder_bvn', rolling_features, 7).iloc[-1]

    holder_window_30 = tracker.rolling_averages(new_holder_df, 'holder', rolling_features, 30).iloc[-1]
    holder_bvn_window_30 = tracker.rolling_averages(new_holder_bvn_df, 'holder_bvn', rolling_features, 30).iloc[-1]

    holder_window_120 = tracker.rolling_averages(new_holder_df, 'holder', rolling_features, 120).iloc[-1]
    holder_bvn_window_120 = tracker.rolling_averages(new_holder_bvn_df, 'holder_bvn', rolling_features, 120).iloc[-1]


    data = {**data, **holder_window_1, **holder_bvn_window_1, **holder_window_7, **holder_bvn_window_7, **holder_window_30, **holder_bvn_window_30, **holder_window_120, **holder_bvn_window_120}
    return data


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

        report_transaction = st.checkbox("🔢 Report Transaction")

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
            'reported': report_transaction,
            'kyc': int(account['kyc']),
            'merchant': True
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

        report_transaction = st.checkbox("🔢 Report Transaction")

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
            'reported': report_transaction,
            'kyc': int(account['kyc']),
            'merchant': True
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

        report_transaction = st.checkbox("🔢 Report Transaction", )

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
            'reported': report_transaction,
            'kyc': account['kyc'],
            'merchant': account['merchant']
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



