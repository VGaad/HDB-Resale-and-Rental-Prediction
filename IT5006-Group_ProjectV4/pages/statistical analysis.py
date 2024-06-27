import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import pydeck as pdk
import numpy as np
import matplotlib.cm as cm
import folium
from streamlit_folium import st_folium
from folium.plugins import FloatImage

# streamlit run pages/streamlit0223_v1.py --server.maxMessageSize 500

st.markdown("# Rental and Resale flat price analysis for HDB in Singapore")

def plot_pricepersqm(data_yr,select_town):
    data_yr['price_per_sqm'] = data_yr[y_axis] / data_yr['floor_area_sqm']
    town_filter = data_yr[data_yr['town'] == select_town ]
    avg_price_per_sqm = data_yr.groupby('Year')['price_per_sqm'].mean()
    town_filter_price_per_sqm = town_filter.groupby('Year')['price_per_sqm'].mean()

    col1, col2, col3 = st.columns([3, 1, 1])

    with col1:
        fig, ax = plt.subplots(figsize=(8, 5))
        avg_price_per_sqm.plot(marker='o', color='blue', label='All Towns')
        town_filter_price_per_sqm.plot(marker='o',color='red', label=select_town)
        plt.title(f'Average Prices per Square Meter in {select_town} versus All Towns in Singapore')
        plt.xlabel('Year')
        plt.ylabel('Average Price per Square Meter')
        plt.grid(True)
        plt.tight_layout()
        plt.legend()
        st.pyplot(fig)

    with col2:
        min_value = round(avg_price_per_sqm.min(), 2)
        max_value = round(avg_price_per_sqm.max(), 2)
        delta = max_value - min_value
        delta_percent = (delta / min_value) * 100  # Calculate delta as a percentage
        st.metric(label="- All Towns Min:", value=f"{min_value:.2f}")
        st.metric(label="- All Towns Max:", value=f"{max_value:.2f}", delta=f"{delta_percent:.2f}%")
    with col3:
        town_min_value = round(town_filter_price_per_sqm.min(), 2)
        town_max_value = round(town_filter_price_per_sqm.max(), 2)
        town_delta = town_max_value - town_min_value
        town_delta_percent = (town_delta / town_min_value) * 100  # Calculate delta as a percentage
        st.metric(label="- Town Min:", value=f"{town_min_value:.2f}")
        st.metric(label="- Town Max:", value=f"{town_max_value:.2f}", delta=f"{town_delta_percent:.2f}%")

def plot(x_axis, y_axis, data, select_year):
    filtered_df = data[data['Year'] == select_year]

    # 0222
    months_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"] #Added Months
    filtered_df['Month Name'] = pd.Categorical(filtered_df['Month Name'], categories=months_order, ordered=True) #Converting the month name column to categorical
    filtered_df.sort_values(by='Month Name', inplace=True) #Sorting the DataFrame by month

    average_prices = filtered_df.groupby(x_axis)[y_axis].mean().reset_index()
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.lineplot(x=x_axis, y=y_axis, data=average_prices, marker='o', markersize=8, color='blue')
    plt.title(f'Average Resale Prices for Year {select_year}')
    plt.xlabel('Month')
    plt.ylabel(f'Average {y_axis}')
    plt.grid(True)
    plt.tight_layout()
    st.pyplot(fig)

def bar_plot_chart(data,select_town,min_year,max_year, y_axis):
    data = data[(data['Year'] >= min_year) & (data['Year'] <= max_year)]
    data = data[data['town'] == select_town]
    flat_types = np.sort(data['flat_type'].unique())
    years = data['Year'].unique()
    mean_values = {}
    for year in years:
        year_data = data[data['Year'] == year]
        mean_values[year] = [year_data[year_data['flat_type'] == flat_type][y_axis].mean() for flat_type in flat_types]

    mean_values_df = pd.DataFrame(mean_values, index=flat_types)

    fig, ax = plt.subplots(figsize=(10, 6))
    mean_values_df.plot(kind='bar', ax=ax)
    ax.set_title('Average Price Variation by Flat Type and Year')
    ax.set_xlabel('Flat Type')
    ax.set_ylabel('Average Price')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    st.pyplot(fig)

def street_vs_meanprice(data,select_town,min_year,max_year, y_axis):
    data = data[(data['Year'] >= min_year) & (data['Year'] <= max_year)]
    data = data[data['town'] == select_town]
    street_price_mean = data.groupby('street_name')[y_axis].mean().sort_values()

    fig, ax = plt.subplots(figsize=(10, 6))
    street_price_mean.plot(kind='bar', color='skyblue', ax=ax)
    ax.set_title(f'Price Variation per Street in {select_town}')
    ax.set_xlabel('Street Name')
    ax.set_ylabel('Average Price')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    for i in range(len(street_price_mean) - 1):
        ax.plot([i, i + 1], [street_price_mean[i], street_price_mean[i + 1]], linestyle='--', color='red')
    initial_price = street_price_mean.iloc[0]
    final_price = street_price_mean.iloc[-1]
    price_change_percentage = ((final_price - initial_price) / initial_price) * 100

    # Display delta as text on the plot
    ax.text(len(street_price_mean) - 1, final_price, f'Delta: {price_change_percentage:.2f}%',
            ha='right', va='bottom', color='red')
    plt.tight_layout()
    st.pyplot(fig)


def lineplot_years(data,select_town,min_year,max_year,y_axis):
    data = data[(data['Year'] >= min_year) & (data['Year'] <= max_year)]

    months_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    data['Month Name'] = pd.Categorical(data['Month Name'], categories=months_order, ordered=True)
    data.sort_values(by=['Year', 'Month Name'], inplace=True) 


    fig4, ax4 = plt.subplots(figsize=(16, 8))
    colors = cm.viridis(np.linspace(0, 1, max_year - min_year + 1))

    for i,year in enumerate(range(min_year, max_year + 1)):
        year_data = data[(data['town'] == select_town) & (data['Year'] == year)]
        avg_price = year_data.groupby('Month Name')[y_axis].mean().reset_index()

        ax4.plot(avg_price['Month Name'], avg_price[y_axis], label=str(year),marker='o', color=colors[i])

    ax4.set_xlabel('Month')
    ax4.set_ylabel('Average Price')
    ax4.set_title('Price Variation Across Months')

    if max_year - min_year > 0:
        ax4.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    st.pyplot(fig4)

def popular_neighbourhood(popular_neighborhoods,min_year,max_year):
    total_transactions = popular_neighborhoods['Transaction Count'].sum()
    col1, col2 = st.columns([1, 3])

    with col1:
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.metric(label="Total Transactions", value=total_transactions)

    with col2:
        fig2,ax2 = plt.subplots(figsize=(16, 8))
        sns.barplot(x='Transaction Count', y='Town', data=popular_neighborhoods, color='blue')
        plt.title(f'Popular Neighborhoods from {min_year}-{max_year} ({dataset_type})')
        plt.xlabel('Transaction Count')
        plt.ylabel('Town')
        plt.grid(True)
        plt.tight_layout()
        st.pyplot(fig2)

def display_map(df, min_year, max_year):
    df = df[(df['Year'] >= min_year) & (df['Year'] <= max_year)]
    map = folium.Map(location = [1.3521, 103.8198], zoom_start=11, scrollWheelZoom=False, tiles='cartoDB positron')
    average_prices = df.groupby('town')[y_axis].mean().reset_index()

    choropleth = folium.Choropleth(
        geo_data ="datasets/SG_map_data.geojson",
        data=data,
        columns=('town',y_axis),
        key_on= 'feature.properties.name',
        fill_color='YlGnBu',
        line_opacity=0.8,

        highlight = True
        
    )
    choropleth.geojson.add_to(map)

    for feature in choropleth.geojson.data['features']:
        town = feature['properties']['name']
        average_price = average_prices[average_prices['town'] == town][y_axis].values
        if len(average_price)>0:
            average = average_price[0]
            feature['properties']['average price']= "avg_" + y_axis +":"+ str(f'{average:.2f}')
        else:
            feature['properties']['average price'] = y_axis + "No data"

    choropleth.geojson.add_child(
        folium.features.GeoJsonTooltip(['name','average price'], labels=False)
    )

    st_map = st_folium(map, width=700, height=450)

    min_val = df[y_axis].min()
    max_val = df[y_axis].max()

    fig, ax = plt.subplots(figsize=(6, 1))
    fig.subplots_adjust(bottom=0.5)

    cmap = mpl.cm.YlGnBu
    norm = mpl.colors.Normalize(vmin=min_val, vmax=max_val)

    cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                norm=norm,
                                orientation='horizontal')
    cb1.set_label('Average Price')
    
    st.pyplot(fig)

# Load the data
dataset_type = st.sidebar.radio('Select Dataset Type', ['Resale', 'Rental'])

if dataset_type == 'Resale':
    data = pd.read_csv("datasets/geocoded_data_resale.csv")
    x_axis = 'Month Name'
    y_axis = 'resale_price'
    y_label = 'Resale Price' #Added
else:
    data = pd.read_csv("datasets/geocoded_data_rental.csv")
    x_axis = 'Month Name'
    y_axis = 'monthly_rent'
    y_label = 'Monthly Rent' #Added



min_year = data['Year'].min()
max_year = data['Year'].max()

select_year_range = st.sidebar.slider('Select year range', min_year, max_year, (min_year, max_year))

data_yr = data[(data['Year'] >= select_year_range[0]) & (data['Year'] <= select_year_range[1])]

select_town = st.sidebar.selectbox('Select Town', sorted(data_yr['town'].unique()))

popular_neighborhoods = data_yr['town'].value_counts().reset_index()
popular_neighborhoods.columns = ['Town', 'Transaction Count']


if dataset_type == 'Resale' and select_year_range[1]-select_year_range[0]>0 and select_year_range[0]!=2024:
    plot_pricepersqm(data_yr,select_town)
popular_neighbourhood(popular_neighborhoods,select_year_range[0], select_year_range[1])

if select_year_range[0]!=2024:
    lineplot_years(data,select_town,select_year_range[0],select_year_range[1],y_axis)
street_vs_meanprice(data,select_town,select_year_range[0],select_year_range[1], y_axis)
bar_plot_chart(data,select_town, select_year_range[0],select_year_range[1], y_axis)
display_map(data,select_year_range[0],select_year_range[0] )
