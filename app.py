# Import Packages
import os
import json
import requests
from dotenv import load_dotenv
from pathlib import Path

# Data
import numpy as np
import pandas as pd

# Visualizations
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.express as px

#-----------APP CONFIG---------------#
APP_NAME = 'Churn Prediction App'
st.set_page_config(page_title=APP_NAME,
                   page_icon=':bar_chart:', 
                   layout='wide') #responsive layout
st.title(APP_NAME)
st.markdown(
    '''
    This app predicts whether a given customer is a Churn Customer 
    or a Staying Customer along with other actionable insights
    '''
)
cache_path = Path('./Data/cache/__df__.csv')

#-----------Hide Styles---------------#
unwanted_styles = """ 
<style>
/*Top Right Hamburger icon */
#MainMenu {visibility: hidden;}
/* Header */
header {visibility: hidden;}
/* Footer */
footer {visibility: hidden;}
</style>
"""
st.markdown(unwanted_styles, unsafe_allow_html=True)

#------------Mapbox API--------------#

# Read the Mapbox API key
load_dotenv()
map_box_api = os.getenv("mapbox")

# Set the mapbox access token
px.set_mapbox_access_token(map_box_api)

#------------Sessions--------------#

# APP State
if 'state' not in st.session_state:
    st.session_state['state'] = 'begin'

# Upload State
if 'upload' not in st.session_state:
    st.session_state['upload'] = False

#------------Helper Functions--------------#

# Save Data with predictions
def save_data(df):
    open(cache_path, 'w').write(df.to_csv(index=False))

# Load Data with predictions
def load_data():
    return pd.read_csv(cache_path, 
                    encoding='UTF-8',
                    index_col=False)

#------------Plot Functions--------------#

# Customer Location Map 
def churn_map(df):
    churn_map = px.scatter_mapbox(
            df,
            lat='lat',
            lon='long',
            color='prediction',
            size='monthly_charges',
            color_continuous_scale=px.colors.sequential.Rainbow,
            size_max=25,
            zoom=8)

    churn_map.update_traces(marker=dict(size=np.where(df['prediction'] == 'Churn Customer', 25, 8)),
                                    selector=dict(mode='markers'))

    st.plotly_chart(churn_map, use_container_width=True)

# Customer Insights
def customer_analytics(df):
    col1, col2, col3 = st.columns(3)
    contract_details = df['contract'].value_counts()
    internet_details = df['internet_service'].value_counts()
    payment_details = df['payment_method'].value_counts()

    with col1:
        labels = contract_details.index.tolist()
        values = contract_details.values.tolist()
        fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.5)])
        fig.update_layout(title_text="<b>Contract Type</b>")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        labels = internet_details.index.tolist()
        values = internet_details.values.tolist()
        fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.5)])
        fig.update_layout(title_text="<b>Internet Service</b>")
        st.plotly_chart(fig, use_container_width=True)
    with col3:
        labels = payment_details.index.tolist()
        values = payment_details.values.tolist()
        fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.5)])
        fig.update_layout(title_text="<b>Payment Method</b>")
        st.plotly_chart(fig, use_container_width=True)

#----------SIDEBAR----------------#

# Sidebar Header
st.sidebar.header('Upload your EXCEL / CSV data')

# Loading Dataset Module
upload_data = st.sidebar.file_uploader("Upload your customers data here..", type=['csv', 'xlsx'])

if upload_data is not None:
    st.session_state['upload'] = True
else:
    st.session_state['upload'] = False


if st.session_state['upload']:

    # Custom Header
    st.subheader('Customer Data')

    # Get Uploaded Data as a DataFrame
    def get_data():
        if st.session_state['state'] == 'begin':
            if upload_data is not None:
                # Read the dataset
                ext = upload_data.name.rsplit('.', 1)[1]
                if ext == 'csv':
                    df = pd.read_csv(upload_data, 
                                        encoding='UTF-8',
                                        index_col=False)
                elif ext == 'xlsx':
                    df = pd.read_excel(upload_data,
                                        sheet_name=0,
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
                st.info('Upload your CSV / Excel file from the sidebar section')
                return None
        
        else:
            # If prediction is already made then load the Dataset with prediction
            return load_data()

    # Load the data
    telco_df = get_data()

    #-----------BODY---------------#

    # Machine Learning Prediction through API
    if st.session_state['state'] == 'begin':
        if st.button("Predict"):
            # Checking if the data is available
            if upload_data is not None:
                # Chage the session state to predict
                st.session_state['state'] = 'predict'
                # Loading Bar Begin
                my_bar = st.progress(0)
                
                predictions = []
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
                    
                    # Request a prediction from the API for a customer
                    res = requests.post(
                        "http://ec2-3-25-148-140.ap-southeast-2.compute.amazonaws.com:8000/predict/", # AWS End Point
                        json=json.loads(data[1:-1])) # Pass Customer Data in JSON format
                    # Read the Response as a JSON Format
                    prediction = json.loads(res.text)
                    # Store each prediction in a list
                    predictions.append(prediction['prediction'][0])
                    # Loading Bar
                    my_bar.progress(i+1)

                # Convert predictions to a DataFrame
                prediction_df = pd.DataFrame({'prediction' : predictions})
                # Append it to loaded dataset
                telco_df = pd.concat([prediction_df, telco_df], axis=1)
                # Save the dataset
                save_data(telco_df)

    # If the data is already predicted
    if st.session_state['state'] == 'predict':

        # Select Churn
        churn = st.sidebar.multiselect(
            "Churn",
            options=telco_df['prediction'].unique(),
            default=telco_df['prediction'].unique()
        )
        
        # Select Gender
        gender = st.sidebar.multiselect(
            "Gender",
            options=telco_df['gender'].unique(),
            default=telco_df['gender'].unique()
        )

        # Select Contract
        contract = st.sidebar.multiselect(
            "Contract",
            options=telco_df['contract'].unique(),
            default=telco_df['contract'].unique()
        )

        # Select Internet Service
        internet_service = st.sidebar.multiselect(
            "Internet Service",
            options=telco_df['internet_service'].unique(),
            default=telco_df['internet_service'].unique()
        )

        # Select Payment Method
        payment_method = st.sidebar.multiselect(
            "Payment Method",
            options=telco_df['payment_method'].unique(),
            default=telco_df['payment_method'].unique()
        )

        # Select City
        city = st.sidebar.multiselect(
            "City",
            options=telco_df['city'].unique(),
            default=telco_df['city'].unique()
        )

        # Filtering Data
        telco_df_query = telco_df.query(
            '''
            gender == @gender & \
            prediction == @churn & \
            contract == @contract & \
            internet_service == @internet_service & \
            payment_method == @payment_method & \
            city == @city
            '''
        )

        # Query out based on the filters
        telco_df = telco_df_query.reset_index(drop=True)
        st.dataframe(telco_df)
        st.caption(f"Rows: {telco_df.shape[0]} | Columns: {telco_df.shape[1]}")

        # Customers Location MAP
        st.subheader('Churn Map')
        churn_map(telco_df)

        # Customer Analytics
        st.subheader('Customer Analytics')
        customer_analytics(telco_df)

else:
    # Check if the cache data is available and remove before restart
    try:
        if cache_path.is_file():
            os.remove(cache_path)
    except OSError as e:
        print(f"Error log: {e.filename} : {e.strerror}")

    # Reset the session
    st.session_state['state'] = 'begin'
    # Upload notification to begin
    st.info('Upload your CSV / Excel file from the sidebar section')