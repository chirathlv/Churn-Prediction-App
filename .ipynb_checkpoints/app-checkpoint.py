# Import Packages
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import time

#--------------------------#

# APP CONFIG
APP_NAME = 'Churn Prediction App'
st.set_page_config(page_title=APP_NAME,
                   page_icon=':bar_chart:', 
                   layout='wide') #responsive layout
st.title(APP_NAME)
st.markdown('This app predicts for a given customer whether he will leave the company or staying')

#--------------------------#

# SIDEBAR

# Loading Dataset Module
with st.sidebar.header('Upload your EXCEL / CSV data'):
    upload_data = st.sidebar.file_uploader("Upload your customers data here..", type=['csv', 'xlsx'])

#--------------------------#

# BODY

# Display Uploaded Dataset
st.subheader('Customer Data')

if upload_data is not None:

    # Read the dataset
    ext = upload_data.name.rsplit('.', 1)[1]
    if ext == 'csv':
        telco_df = pd.read_csv(upload_data, 
                               encoding='UTF-8')
    elif ext == 'xlsx':
        telco_df = pd.read_excel(upload_data,
                                 engine='openpyxl', 
                                 sheet_name=0, 
                                 encoding='UTF-8')
    else:
        telco_df = None

    if (telco_df is not None):
        if not telco_df.empty:
            st.write(telco_df)
            st.caption(f"Rows: {telco_df.shape[0]} | Columns: {telco_df.shape[1]}")
        else:
            st.info('Your Dataset is empty, please check if you have data in the file before uploading!')
    else:
        st.info('Your Dataset has a problem, please check before uploading!')
else:
    st.info('Upload your CSV file from the sidebar section')


# Machine Learning Prediction

