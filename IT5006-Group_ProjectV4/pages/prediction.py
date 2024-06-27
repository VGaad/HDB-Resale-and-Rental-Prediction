import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pickle
import xgboost as xgb
import altair as alt
from matplotlib.ticker import FuncFormatter


def lineplot_townprice(loaded_model, X_pred, encoded_value, selected_month, selected_month_spi, dataset_type):
    dd = {}
    for i in range(len(town_list)):
        one_hot_encoded[i] = 1
        features = [selected_month, 2024, selected_month_spi]
        if dataset_type == 'Resale':
            features.extend([input_floor_area_sqm, input_remain_lease, encoded_storeyrange])
        features.extend(one_hot_encoded + [encoded_value])
        X_pred = np.array(features)
        X_pred = X_pred.reshape(1, -1)
        prediction = loaded_model.predict(X_pred)
        dd[town_list[i]] = prediction
        one_hot_encoded[i] = 0

    df = pd.DataFrame(list(dd.items()), columns=['Town', 'Predicted Price'])

    fig4, ax4 = plt.subplots(figsize=(16, 8))

    ax4.plot(df['Town'], df['Predicted Price'], label='plot', marker='o')
    ax4.set_xticklabels(df['Town'], rotation=90, ha='right')
    ax4.set_xlabel('Towns')
    ax4.set_ylabel('Predicted rental Price')
    ax4.set_title('Predicted price across town')
    # ax4.legend()
    # st.pyplot(fig4)
    # 设置y轴的格式为普通数字格式
    ax4.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:,.0f}'))

    ax4.legend()
    st.pyplot(fig4)


def show_prediction(loaded_model,X_pred,std_error):
    prediction = loaded_model.predict(X_pred)
    rounded_prediction = int(round(prediction[0]))
    # st.metric(label="Price Prediction", value=rounded_prediction)
    st.markdown(f"""
    <div style="padding: 1em; border-radius: 0.25em; border: 1px solid #eee; margin: 1em 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
        <h1 style="color: #2678F9; font-size: 1.5em; font-weight: 600; margin: 0; padding: 0;">Predicted Price</h1>
        <p style="color: #2678F9; margin: 0.25em 0 0; font-size: 2em; font-weight: bold;">SGD$ {rounded_prediction}</p>
        <p style="color: #000; margin: 0;"><strong>in {selected_town}</strong> for a<strong> {selected_flat_type}</strong></p>
    </div>
    """, unsafe_allow_html=True)
    lower_bound = prediction[0] - 1.96 * std_error
    upper_bound = prediction[0] + 1.96 * std_error
    st.metric(label=r"Range of price prediction (with 95% confidence)", value=f"({int(round(lower_bound))}, {int(round(upper_bound))})")


def prediction_plot_months(loaded_model, X_pred, one_hot_encoded, Months, dataset_type):
    dm = {}
    for i, month in enumerate(Months):
        selected_month = i + 4
        selected_month_spi = data_spi[selected_month]
        features = [selected_month, 2024, selected_month_spi]
        if dataset_type == 'Resale':
            features.extend([input_floor_area_sqm, input_remain_lease, encoded_storeyrange])
        features.extend(one_hot_encoded + [encoded_value])
        X_pred = np.array(features)
        X_pred = X_pred.reshape(1, -1)
        prediction = loaded_model.predict(X_pred)
        dm[month] = prediction

    df = pd.DataFrame(list(dm.items()), columns=['Months', 'Predicted Price'])
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.plot(df['Months'], df['Predicted Price'], label='plot', marker='o')
    ax.set_xticklabels(df['Months'], rotation=90, ha='right')

    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:,.0f}'))

    ax.set_xlabel('Months')
    ax.set_ylabel('Predicted rental Price')
    ax.set_title('Predicted price across Months')
    ax.legend()
    st.pyplot(fig)



st.markdown('## House price prediction')

dataset_type = st.sidebar.radio('Select Dataset Type', ['Resale', 'Rental'])

if dataset_type == 'Resale':
    loaded_model = pickle.load(open("datasets/resale.pickle.dat", "rb"))
    std_error = 35196.363
    data = pd.read_csv("datasets/resale_spi_encoded.csv")
    data_spi ={
        4: 188.09729,
        5: 188.0977,
        6: 188.09694,
        7: 188.0967,
        8: 188.10057,
        9: 188.08748,
        10: 188.06815,
        11: 188.06816,
        12: 188.07208
    }

    storey_range = [
        '01 TO 05',
        '06 TO 10',
        '11 TO 15',
        '16 TO 20',
        '21 TO 25',
        '26 TO 30',
        '31 TO 35',
        '36 TO 40',
        '41 TO 45',
        '46 TO 51',
    ]

    storey_range_mapping = {
        '01 TO 05':1,
        '06 TO 10':2,
        '11 TO 15':3,
        '16 TO 20':4,
        '21 TO 25':5,
        '26 TO 30':6,
        '31 TO 35':7,
        '36 TO 40':8,
        '41 TO 45':9,
        '46 TO 51':10,
    }

    flat_type_mapping = {
        '1_ROOM': 1,
        '2_ROOM': 2,
        '3_ROOM': 3,
        '4_ROOM': 4,
        '5_ROOM': 5,
        'EXECUTIVE': 6,
        'MULTI_GENERATION': 7
    }
    input_floor_area_sqm = st.sidebar.slider('Select Floor Area (sqm)', min_value=20, max_value=250, value=100)

    input_remain_lease = st.sidebar.slider('Select Remaining Lease', min_value=10, max_value=94, value=50)

    selected_story_range = st.sidebar.selectbox('Select Story_range',storey_range )

    town_list = ['ANG MO KIO', 'BEDOK', 'BISHAN', 'BUKIT BATOK', 'BUKIT MERAH',
       'BUKIT PANJANG', 'BUKIT TIMAH', 'CENTRAL AREA', 'CHOA CHU KANG',
       'CLEMENTI', 'GEYLANG', 'HOUGANG', 'JURONG EAST', 'JURONG WEST',
       'KALLANG/WHAMPOA', 'MARINE PARADE', 'PASIR RIS', 'PUNGGOL',
       'QUEENSTOWN', 'SEMBAWANG', 'SENGKANG', 'SERANGOON', 'TAMPINES',
       'TOA PAYOH', 'WOODLANDS', 'YISHUN']

    flat_type = ['1_ROOM', '2_ROOM','3_ROOM', '4_ROOM', '5_ROOM', 'EXECUTIVE',
       'MULTI_GENERATION']

    flat_type_mapping = {
        '1_ROOM': 1,
        '2_ROOM': 2,
        '3_ROOM': 3,
        '4_ROOM': 4,
        '5_ROOM': 5,
        'EXECUTIVE': 6,
        'MULTI_GENERATION':7
    }

    one_hot_encoded = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

else:
    loaded_model = pickle.load(open("datasets/rental.pickle.dat", "rb"))
    std_error = 496.642
    data = pd.read_csv("datasets/rental_spi_encoded.csv")
    data_spi = {
        4: 135.4204,
        5: 134.82086,
        6: 134.82599,
        7: 132.02582,
        8: 130.33408,
        9: 127.05162,
        10: 127.04671,
        11: 127.049835,
        12: 127.05638
    }

    flat_type = ['1-ROOM','2-ROOM','3-ROOM','4-ROOM','5-ROOM','EXECUTIVE']

    flat_type_mapping = {
        '1-ROOM': 1,
        '2-ROOM': 2,
        '3-ROOM': 3,
        '4-ROOM': 4,
        '5-ROOM': 5,
        'EXECUTIVE': 6
    }

    town_list = ['ANG MO KIO', 'BEDOK', 'BISHAN', 'BUKIT BATOK', 'BUKIT MERAH', 'BUKIT PANJANG', 'BUKIT TIMAH',
                 'CENTRAL', 'CHOA CHU KANG', 'CLEMENTI', 'GEYLANG',
                 'HOUGANG', 'JURONG EAST', 'JURONG WEST', 'KALLANG/WHAMPOA', 'MARINE PARADE', 'PASIR RIS', 'PUNGGOL',
                 'QUEENSTOWN', 'SEMBAWANG', 'SENGKANG',
                 'SERANGOON', 'TAMPINES', 'TOA PAYOH', 'WOODLANDS', 'YISHUN']

    flat_type = ['1-ROOM', '2-ROOM','3-ROOM', '4-ROOM', '5-ROOM', 'EXECUTIVE']
    one_hot_encoded = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


Months = ['April','May','June','July','August','September','October','November','December']
select_month = st.sidebar.selectbox('Select Month',Months)

selected_town = st.sidebar.selectbox('Select Town', town_list)

selected_flat_type = st.sidebar.selectbox('Select Flat Type', flat_type)

selectedtown_index = town_list.index(selected_town)

one_hot_encoded = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
one_hot_encoded[selectedtown_index] = 1

selected_month = 4 + Months.index(select_month)
selected_month_spi = data_spi[selected_month]

encoded_value = flat_type_mapping[selected_flat_type]

if dataset_type == 'Resale':
    encoded_storeyrange = storey_range_mapping[selected_story_range]
    X_pred = np.array([selected_month, 2024, selected_month_spi,input_floor_area_sqm,input_remain_lease,encoded_storeyrange] + one_hot_encoded+[encoded_value])
    X_pred = X_pred.reshape(1, -1)
    st.write("In Resale")
    show_prediction(loaded_model, X_pred,std_error)
    prediction_plot_months(loaded_model, X_pred, one_hot_encoded, Months, 'Resale')
    lineplot_townprice(loaded_model, X_pred, encoded_value, selected_month, selected_month_spi, 'Resale')

elif dataset_type == 'Rental':
    X_pred = np.array([selected_month, 2024, selected_month_spi] + one_hot_encoded+[encoded_value])
    X_pred = X_pred.reshape(1, -1)

    show_prediction(loaded_model,X_pred,std_error)
    prediction_plot_months(loaded_model, X_pred, one_hot_encoded, Months, 'Rental')
    lineplot_townprice(loaded_model, X_pred, encoded_value, selected_month, selected_month_spi, 'Rental')
