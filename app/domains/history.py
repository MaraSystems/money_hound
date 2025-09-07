import streamlit as st
from .dataframe import list_states, list_accounts, get_account, select_location, list_devices, get_transactions, classified_transactions_df
from sklearn.preprocessing import RobustScaler, LabelEncoder

def main():
    st.title("Transaction Histories")
    st.set_page_config(layout="wide")

    df = classified_transactions_df[['amount', 'balance', 'holder', 'related', 'category', 'type', 'channel', 'device', 'reported', 'holder_bvn', 'related_bvn', 'related_bank', 'fraud', 'fraud_score']]
    st.write(df)