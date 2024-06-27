import streamlit as st
import random

st.markdown('''
# Welcome to the HDB Rental Price Prediction Dashboard

Are you planning on buying the resale flat or looking for rental property ? 
No worries! You can view the past statistical trends and also predictions with 95% confidence for your requirement.
        


Team 3: 
            

''')
names = ["Cheng Xianda", "Sai Niharika Naidu Gandham", "Vaishnavi Gaad", "Tay See Boon"]

random.shuffle(names)

for name in names:
    st.markdown(f"#### **{name}**")





# ### Dataset and Preprocessing
# 1. Resale Dataset Summary 
# This dataset details HDB resale transactions in 27 Singapore towns from 2019 to 2023. It records the town, flat type, block, street name, storey range, floor area, flat model, lease start date, and resale price. 

# 2. Rental Dataset Summary 
# The rental dataset covers HDB transactions from 2021 to 2023 across the 26 towns. Key information includes rent approval status, town, block, street, flat type, and monthly rental cost. 

# 3. Singapore Real Estate Exchange Property Index (SPI) 
# SPI is the first index to calculate price changes that take into account unique Singaporean factors such as the property's distance to a top primary school or an MRT station. The index, of course, controls standard index factors like location, age of property, size, floor levels and land tenure. 

# SPI uses a Hedonic Regression methodology modeled on proven real estate economics and consumer price indices worldwide. 

# Flowchart of Data preprocessing and Machine learning process:

# st.image('images/flowchart.png', caption='Data Preprocessing and Machine Learning Process Flowchart')

# st.markdown('''
# ### Data preprocessing 
# The preprocessing of the dataset comprised two main stages: Data Cleaning and data Transformation
# 1.	Data Cleaning 

# This phase involves preparing the dataset by addressing issues that might be inaccurate in the analysis, such as inconsistencies. This includes standardizing column names and values to eliminate inconsistencies. Our team has checked for and removed any duplicate datapoint. For instance, any spaces between words were substituted with an underscore "_" to achieve uniformity across the in the flat types and models, "2 ROOM” à “2_ROOM"

# Checking the dataset for any missing values and performing imputation on these values enhances the integrity and accuracy of the data for. For example, the 'Remaining Lease' variable is missing over 50% of its data, which significantly influences resale prices. To address this, we investigated the official HDB website and applied their 99-year lease policy. By employing the formula [Remaining Lease Term = 99 years - (Current Year - Lease Start Year)], we can accurately impute these important missing values.

# 2.	Feature selection
            
# Our team has removed columns “block”, “street name”, “flat mode” and “lease start date” due to the redundancy and reduce the noise while training the models.  We observed that block and street name shared common characteristics with town and do not contribute additional information, they may be considered redundant. For instance, the town was "Bishan" whenever the street name contains "Bishan," then having both might not provide extra information useful for the model's predictions. Removing redundant features can help in simplifying the model without compromising its performance. Similarly, the flat model shared common attributes with regards to floor area; for example, from 2008 onward, the 3-Room Model A was designated as having 65 square meters. Lease start date removed after new feature was created based on its information.

# 3.	Feature creation
            
# To deepen the understanding of the dataset, a new feature, "Average Price Per Square Meter," was introduced. This was calculated using the formula [Average Price Per Square Meter = Resale Price / Flat Area], providing users with a transparent view of the cost for each square meter.

# 4.	Data Transformation 

# This stage involves adjusting the dataset to make it suitable and efficient for use. The transformations included:

# Data Consolidation: Combining five separate resale files into a single, unified dataset.

# Date Transformation: Converting date information into datetime format and segregating it into 'Year' and 'Month' segments for enhanced temporal analysis.

# One Hot Encoding: The "Town" variable in both the resale and rental datasets was converted into a one-hot encoded format to facilitate model training.

# Binning and Ordinal Encoding: The "Storey Range" variable, which consists of various ranges was categorized into 10 bins with intervals of 5 levels, ranging from 01 TO 05 up to 46 TO 51. Subsequently, ordinal encoding was applied to optimize the data for training purposes.
# ''')

# st.markdown('''
# ## Exploratory Data Analysis

# **Average Prices per Square Meter versus Year:** 
# For all Singapore towns compared together, this line graph showed an increasing trend from 1990 to 1997. Decline in resale prices was observed from 1997 to 1999, followed by somewhat stagnant prices till 2006. A sharp increase was observed from 2006 till 2013. From 2013 a declining trend was observed with a slight increase in 2015, only to decline again in 2018 till 2020. The prices boomed post 2020.

# The decrease in resale prices in the year 1997 is explainable by the ‘Asian financial crisis (1997–1998)’ which started in July 1997 in Thailand with the collapse of Thai Baht, soon affecting other regional countries. As a result, Singapore experienced recession in the later half of 1998 ([source](https://www.nlb.gov.sg/main/article-detail?cmsuuid=6a94eaac-75ec-41ff-b5ef-38154ccae4e0)).

# The declining trend observed in 2013 and then in 2018 can be attributed to the ‘Property Cooling Measures’ implemented by Singapore government. This measure intended to ensure that flats remain affordable ([source](https://www.channelnewsasia.com/singapore/property-cooling-measures-hdb-resale-prices-2013-2018-each-singapore-town-2385831)).

# **Popular Neighborhoods:** 
# The number of resale and rental prices transactions are highest for Tampines, Yishun, Woodlands, Bedok, Jurong West, Sengkang towns. This is consistent with the HDB statistics. These towns being among the towns having maximum number of dwelling units ([source](https://assets.hdb.gov.sg/about-us/news-and-publications/annual-report/2022/ebooks/Key%20Statistics%20FY21.pdf)).

# **Price Variations Across Months:** 
# In most years, the prices seem to be more or less steady across months in a year for both resale and rental. Fluctuations however, are observed during special times such as during ‘Asian Financial Crisis’ and ‘COVID pandemic’.

# **Street-wise Price Variations:** 
# On a general note observed across all Singapore towns, HDBs near Metros, shopping centers, prime office locations are more expensive than other areas. 

# **Average Price Variations by Flat Type and Year:** 
# 2 Room flats price the least, followed by 3 Room, 4 Room, 5 Room, and Executive. In case resale across years, each room type mirror the trend observed in ‘Average Prices per Square Meter versus Year’ visualization.

# **Geospatial Distribution of Average Prices across Towns:** 
# For HDB resale, Queenstown, Bishan, Ang Mo Kio towns have highest average prices and are coloured the darkest on choropleth map.

# In case of rental prices, Jurong West, Jurong East, Bukit Batok, Queenstown, Bishan, Kallang/Whampoa, Pasir Ris, Sembawang towns have high average rental prices.

# Queenstown, Singapore’s first satellite town, is very well connected to the rest of Singapore via multiple MRT stations on two MRT lines – East West line and Central line. The town also has multiple bus routes, 3 bus terminals and is also connected to major expressways - Central Expressway (CTE) and Ayer Rajah Expressway (AYE). Queenstown also has National University of Singapore. Connectivity, a variety of dwelling options, recreational activities are major reasons in increased resale and rental prices in this town ([source](https://dollarsandsense.sg/neighbourhood-estate-guide-queenstown-crown-jewel-housing-estates-singapore/)).

# In case of Bishan, there are plenty of schools in the area. It also has MRT connectivity through North-South line and Circle line. Bishan’s Sin Ming neighborhood expanding into a business/industrial hub is another reason for increased HDB rental and resale prices in Bishan ([source](https://www.99.co/singapore/insider/bishan-hdb-resale-flats/)).

# Ang Mo Kio like Bishan has great connectivity, plenty of academic options. It is an all-in-one neighbourhood with numerous markets, food centers spread across the town ([source](https://www.homeanddecor.com.sg/design/news/property-why-ang-mo-kio-is-a-coveted-area-to-live-in/)). Also, Singapore PM holds the MP seat from Ang Mo Kio, making this town first in Singapore to experience latest policy implementations.
# ''')

# st.markdown('''
# ## Models and performance 
# Four machine learning algorithms were employed in this project including logistic regression, decision tree, random forest and extreme gradient boosting (Xgboost) were trained to perform regression classifier task with the goal of generating the prediction of  Singapore HDB resale and rental price.  The train and test split were 80: 20 in both resale and rental dataset. We apply stratified k fold cross validation (k fold = 5) to ensure that the model is trained and validated on varied yet representative samples of the data, leading to better generalization capabilities. The best model for predictive price in both resale and rental was Randon Forest the root mean square error (RMSE) achieved at 490.82. The performance of 4 models is shown in Table 1.

# #### Table 1. Performance of four machine learning models 

# | Model            | Logistic regression | Decision tree | Random forest | XGBoost |
# |------------------|---------------------|---------------|---------------|---------|
# | RMSE             |                     |               |               |         |
# | Resale           |  35196.363          |  491.48       |  490.82       |  491.48 |
# ''')

# st.markdown('''## Reflection and Conclusion
            
# In our project, our team encountered several challenges. Both were in understanding the specific domain and in the technical aspects of our work.
# One of the challenges was addressing the missing values for the remaining lease. Initially, we lacked knowledge about Singapore's Housing and Development Board (HDB) lease policies. Through thorough research on the official website of HDB, we gained knowledge of the 99-year lease policy. This experience underscored an important lesson: data analysts or scientists should not immediately jump into data preprocessing upon receiving a dataset. Being literate in data and possessing domain knowledge is essential for effective performance in real-world situations.
# Furthermore, the process of implementing solutions involves iterative trial and error. It's important to remain calm in the face of error messages. You will reach success by not giving up and trying repeatedly. Specifically, when attempting to load a map within the Streamlit platform, we faced a constraint with the 200 MB size limit. This required us to investigate and select among various map options that would comply with the size restrictions.

# In conclusion, our team has successfully developed a web application tailored for the analysis and prediction of Singapore HDB resale and rental prices. This application offers a comprehensive examination, assisting users with critical insights to make well-informed decisions in their resale or rental housing.

# ''')