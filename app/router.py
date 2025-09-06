import streamlit as st
from .domains import home, history, detect

routes = {
    'HOME': ('Home', home.main),
    'HISTORY': ('History', history.main),
    'DETECT': ('Detect', detect.main)
}

def main(page):
    routes[page][1]()