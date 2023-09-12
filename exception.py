import sys
import streamlit as st
from src.logger import logging
from src.exception import CustomException

st.title('Exception  Testing')

def exception_test():
    logging.info('We are testing exception file')
    try:
        raise Exception('testing exception file')
    
    except Exception as e:
        messege=CustomException(e,sys)
        logging.info(messege.error_message)

if __name__ == '__main__':
    exception_test()
    st.write("exception test succeeded")