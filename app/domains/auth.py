import streamlit as st
from .dataframe import list_accounts, get_account

def login(page):
    st.subheader('Login')

    with st.form('Login'):
        account_id = st.selectbox("🔢 Account No.", list_accounts())
        submitted = st.form_submit_button("Login")

        if submitted:
            account = get_account(account_id)
            st.session_state['account'] = account
            st.session_state['page'] = page


def logout():
    st.session_state['account'] = None
    st.session_state['page'] = 'HOME'

