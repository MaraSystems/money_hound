import streamlit as st
from app.domains import auth
from app import router

event_map = router.routes

def main():
    st.sidebar.title('MoneyHound')
    account = None

    if "account" in st.session_state:
        account = st.session_state['account']
        st.sidebar.write(account['account_no'])

    if "page" not in st.session_state:
        st.session_state['page'] = "HOME"

    for key in event_map.keys():
        if st.sidebar.button(key):
            st.session_state['page'] = key

    router.main(st.session_state['page'])
    

    if "account" in st.session_state:
        logout = st.sidebar.button('Logout')

        if logout:
            auth.logout()

main()