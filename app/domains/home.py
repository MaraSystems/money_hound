import streamlit as st
import pandas as pd
import os

from . import dataframe

def main():
    st.set_page_config(layout="wide")
    menu_tabs = st.tabs(["🏦 Banks", "👤 Users", "🌍 States", 'About'])

    with menu_tabs[0]:
        st.subheader("Banks Dataset")
        st.write(dataframe.bank_devices_df)

        st.subheader("Count of Bank Devices")
        bank_counts = dataframe.bank_devices_df["bank_id"].value_counts()
        st.bar_chart(bank_counts)


    with menu_tabs[1]:
        st.subheader("Users Dashboard")
        st.write("Here you can view user-related analytics.")


    with menu_tabs[2]:
        st.subheader("States Dashboard")
        st.write("Here you can view state-related analytics.")




