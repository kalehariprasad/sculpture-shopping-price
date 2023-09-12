
import streamlit as st
from src.logger import logging

st.title('Logging  Testing')

def logging_test():
    logging.info('We are testing logging file')

if __name__ == '__main__':
    logging_test()
    st.write("Logging test succeeded")