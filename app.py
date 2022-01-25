# Import Packages
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import time
import requests
import json
import plotly.express as px
from dotenv import load_dotenv
import os
import numpy as np
from pathlib import Path

#-----------APP CONFIG---------------#
APP_NAME = 'Churn Prediction App'
st.set_page_config(page_title=APP_NAME,
                   page_icon=':bar_chart:', 
                   layout='wide') #responsive layout
st.title(APP_NAME)
st.markdown('This app predicts for a given customer whether he will leave the company or staying')

#------------Mapbox API--------------#

# Read the Mapbox API key
load_dotenv()
map_box_api = os.getenv("mapbox")

# Set the mapbox access token
px.set_mapbox_access_token(map_box_api)

#------------Helper Functions--------------#

if 'key' not in st.session_state:
    st.session_state['key'] = 'begin'

def save_data(df):
    open(Path('./Data/__df__.csv'), 'w').write(df.to_csv(index=False))

# Draw Map
def churn_map(df):
    churn_map = px.scatter_mapbox(
            df,
            lat='lat',
            lon='long',
            color='prediction',
            size='monthly_charges',
            title='<b>Customer Churn Map</b>',
            color_continuous_scale=px.colors.sequential.Rainbow,
            size_max=25,
            zoom=8)

    churn_map.update_traces(marker=dict(size=np.where(df['prediction'] == 'Churn Customer', 25, 8)),
                                    selector=dict(mode='markers'))

    st.plotly_chart(churn_map, use_container_width=True)

#----------SIDEBAR----------------#

# Sidebar Header
st.sidebar.header('Upload your EXCEL / CSV data')

# Loading Dataset Module
upload_data = st.sidebar.file_uploader("Upload your customers data here..", type=['csv', 'xlsx'])

# Display Uploaded Dataset
st.subheader('Customer Data')

def get_data():
    if st.session_state['key'] == 'begin':
        if upload_data is not None:
            # Read the dataset
            ext = upload_data.name.rsplit('.', 1)[1]
            if ext == 'csv':
                df = pd.read_csv(upload_data, 
                                    encoding='UTF-8',
                                    index_col=False)
            elif ext == 'xlsx':
                df = pd.read_excel(upload_data,
                                        engine='openpyxl', 
                                        sheet_name=0, 
                                        encoding='UTF-8',
                                        index_col=False)
            else:
                return None

            if (df is not None):
                if not df.empty:
                    st.caption(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")
                    st.info('Your Dataset is Uploaded. Hit Predict!')
                    return df
                else:
                    st.info('Your Dataset is empty, please check if you have data in the file before uploading!')
                    return None
            else:
                st.info('Your Dataset has a problem, please check before uploading!')
                return None
        else:
            st.info('Upload your CSV file from the sidebar section')
            return None
    
    else:
        df = pd.read_csv(Path('./Data/__df__.csv'), 
                        encoding='UTF-8',
                        index_col=False)
        return df

# Load the data
telco_df = get_data()

#--------------------------#

# Machine Learning Prediction
if st.session_state.key == 'begin':
    if st.button("Predict"):
        if upload_data is not None:
            st.session_state.key = 'predict'
            predictions = []
            my_bar = st.progress(0)
            for i in range(telco_df.shape[0]):
                data = telco_df[['city',
                            'gender',
                            'senior_citizen',
                            'partner',
                            'dependents',
                            'phone_service',
                            'multiple_lines',
                            'internet_service',
                            'online_security',
                            'online_backup',
                            'device_protection',
                            'tech_support',
                            'streaming_tv',
                            'streaming_movies',
                            'contract',
                            'paperless_billing',
                            'payment_method',
                            'monthly_charges',
                            'total_charges',
                            'tenure',
                            'lat',
                            'long',
                            'zip']][i:i+1].to_json(orient='records')

                res = requests.post("http://ec2-3-25-148-140.ap-southeast-2.compute.amazonaws.com:8000/predict/", json=json.loads(data[1:-1]))
                prediction = json.loads(res.text)
                predictions.append(prediction['prediction'][0])
                my_bar.progress(i+1)

            #st.write(pd.DataFrame({'Prediction' : predictions}))
            prediction_df = pd.DataFrame({'prediction' : predictions})
            telco_df = pd.concat([prediction_df, telco_df], axis=1)
            save_data(telco_df)

if st.session_state.key == 'predict':

    # Churn select
    churn = st.sidebar.multiselect(
        "Churn",
        options=telco_df['prediction'].unique(),
        default=telco_df['prediction'].unique()
    )
    
    # Gender select
    gender = st.sidebar.multiselect(
        "Gender",
        options=telco_df['gender'].unique(),
        default=telco_df['gender'].unique()
    )

    # Contract select
    contract = st.sidebar.multiselect(
        "Contract",
        options=telco_df['contract'].unique(),
        default=telco_df['contract'].unique()
    )

    # City select
    city = st.sidebar.multiselect(
        "City",
        options=telco_df['city'].unique(),
        default=telco_df['city'].unique()
    )

    telco_df_query = telco_df.query(
        'gender == @gender & prediction == @churn & contract == @contract & city == @city'
    )

    telco_df = telco_df_query.reset_index(drop=True)
    st.dataframe(telco_df)

    # Customers Location MAP
    churn_map(telco_df)