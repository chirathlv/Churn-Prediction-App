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

#--------------------------#
# APP CONFIG
APP_NAME = 'Churn Prediction App'
st.set_page_config(page_title=APP_NAME,
                   page_icon=':bar_chart:', 
                   layout='wide') #responsive layout
st.title(APP_NAME)
st.markdown('This app predicts for a given customer whether he will leave the company or staying')

#--------------------------#

# Read the Mapbox API key
load_dotenv()
map_box_api = os.getenv("mapbox")

# Set the mapbox access token
px.set_mapbox_access_token(map_box_api)

#--------------------------#

# Helper Functions
placeholder = st.empty()
status = st.empty()
def render(df):
    status.info('Your Dataset is Uploaded. Hit Predict!')
    #placeholder.dataframe(df)
    placeholder.write(df)
    placeholder.caption(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")

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
                            encoding='UTF-8',
                            index_col=False)
    elif ext == 'xlsx':
        telco_df = pd.read_excel(upload_data,
                                engine='openpyxl', 
                                sheet_name=0, 
                                encoding='UTF-8',
                                index_col=False)
    else:
        telco_df = None

    if (telco_df is not None):
        if not telco_df.empty:
            render(telco_df)
        else:
            st.info('Your Dataset is empty, please check if you have data in the file before uploading!')
    else:
        st.info('Your Dataset has a problem, please check before uploading!')
else:
    st.info('Upload your CSV file from the sidebar section')


# Machine Learning Prediction
if st.button("Predict"):

    # Change Status
    with status.container():
        status.empty()

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

        res = requests.post("http://127.0.0.1:8000/predict/", json=json.loads(data[1:-1]))
        prediction = json.loads(res.text)
        predictions.append(prediction['prediction'][0])
        my_bar.progress(i+1)

    #st.write(pd.DataFrame({'Prediction' : predictions}))
    prediction_df = pd.DataFrame({'prediction' : predictions})
    telco_df = pd.concat([prediction_df, telco_df], axis=1)

    with placeholder.container():
        # Update the telco_df
        st.write(telco_df)


    customers_map_plot = px.scatter_mapbox(
        telco_df,
        lat='lat',
        lon='long',
        color='prediction',
        size='monthly_charges',
        title='<b>Customer Churn Map</b>',
        color_continuous_scale=px.colors.sequential.Rainbow,
        size_max=25,
        zoom=8
    )

    customers_map_plot.update_traces(marker=dict(size=np.where(telco_df['prediction'] == 'Churn Customer', 25, 8)),
                                    selector=dict(mode='markers'))

    st.plotly_chart(customers_map_plot, use_container_width=True)