import streamlit as st 
import pandas as pd
import pickle
import numpy as np

st.title('Startup-Acquisition-Status-Prediction')



def input_data():
    st.sidebar.title('Input Values')
    uploaded_file = st.sidebar.file_uploader("Choose a csv file")
    if uploaded_file is not None:
        # Can be used wherever a "file-like" object is accepted:
        dataframe = pd.read_csv(uploaded_file)
        st.write(dataframe)
        return dataframe
        
    founded_at = st.sidebar.selectbox(
                        'Founded at',
                        [x for x in range(1901,2014)]
                        )
    funding_rounds = st.sidebar.select_slider(
                        'Funding Rounds',
                        [x for x in range(4)]
                        )

    milestones = st.sidebar.select_slider(label='Milestones',options=[x for x in range(8)])
    relationship = st.sidebar.select_slider(label='Relationship',options=[x for x in range(9)])
    funding_total_usd = st.sidebar.text_input(
                    label="Funding Total USD (use 'nan' for unknown value)",
                    placeholder='value<29220000'
                        )
    active_days = st.sidebar.text_input(
                        label="Active days (use 'nan' for unknown value)"
                        )
    code_category = st.sidebar.selectbox(
                        'Code Category',
                        ['web','games video','network hosting','advertising','enterprise','consulting','mobile',
                                 'health','software','biotech','ecommerce','public relatiions','hardware','search','other']
                        )
    country_code = st.sidebar.selectbox(
                        'Country Code',
                        ['USA', 'GBR', 'IND', 'CAN', 'DEU', 'FRA', 'AUS', 'ESP', 'IRL','other']
                        )
    try:
        if funding_total_usd == '' or funding_total_usd == 'nan':
            funding_total_usd = np.nan
        else:
            funding_total_usd = float(funding_total_usd)
        if active_days == '' or active_days == 'nan':
            active_days = np.nan
        else:
            active_days = float(active_days)
    except:
        st.sidebar.write('Accepts only number or nan')
    features = np.array([founded_at,funding_rounds, funding_total_usd,milestones, 
                                    relationship,active_days,code_category,country_code])
    dataframe = pd.DataFrame(features.reshape(1,8),
                            columns=['founded_at','funding_rounds', 'funding_total_usd','milestones', 
                                    'relationships','activeDays','category_code', 'country_code']
                            )
    st.subheader('Input Dataframe')
    st.write(dataframe)
    return dataframe


model = pickle.load(open('finalized_model.pkl','rb'))
input_df = input_data()
predict = model.predict(input_df)
predict_prob = model.predict_proba(input_df)
st.subheader('Prediction')
status=['Running','Closed']
st.write('The Status of the company is : ',status[predict[0]])
st.subheader('Prediction Probability')
st.write(predict_prob)

