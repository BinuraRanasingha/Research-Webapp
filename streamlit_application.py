from pyngrok import ngrok


import streamlit as st
import tensorflow as tf
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import seaborn as sns
import plotly.express as px

# Load the saved TensorFlow model
model_tf = tf.keras.models.load_model('/content/f_borrowing.h5')

# Load the pickle model for taxation
with open("/content/taxation.pkl", "rb") as file_tax:
    model_tax = pickle.load(file_tax)

# Load the pickle model for domestic borrowings
with open("/content/Domestic_borrowings_best.pkl", "rb") as file_dom:
    model_dom = pickle.load(file_dom)

# Load the pickle model for Money printing
with open("/content/money_printing_model.pkl", "rb") as file_dom:
    model_mon = pickle.load(file_dom)


# Load the pickle model for bebt sustainability
with open("/content/tot_debt_model.pkl", "rb") as file_debt:
    model_debt = pickle.load(file_debt)


# Load the pickle model for bebt sustainability
with open("/content/GDP.pkl", "rb") as file_gdp:
    model_gdp = pickle.load(file_gdp)

#MongoDB connection for access the past dataset in order to make visualizations#
#------------------------------------------------------------------------------#


uri = "mongodb+srv://Research:research123@fip-2023-087.vekow81.mongodb.net/?retryWrites=true&w=majority"

# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))

db = client['test']
collections = db.list_collection_names()
GovernmentDebt_df = db['GovernmentDebt']
MoneyPrintingCollection_df = db['MoneyPrintingCollection']
ForeignBorrowingCollection_df = db['ForeignBorrowingCollection']
TaxationCollection_df = db['TaxationCollection']
DomesticBorrowingCollection_df = db['DomesticBorrowingCollection']

#Retrieving data from mongodb to induvidual dataframes
#df_GovernmentDebt = pd.DataFrame(list(GovernmentDebt_df.find({},{"_id":0})))
df_MoneyPrintingCollection = pd.DataFrame(list(MoneyPrintingCollection_df.find({},{"_id":0})))
df_ForeignBorrowingCollection = pd.DataFrame(list(ForeignBorrowingCollection_df.find({},{"_id":0})))
df_TaxationCollection = pd.DataFrame(list(TaxationCollection_df.find({},{"_id":0})))
df_DomesticBorrowingCollection = pd.DataFrame(list(DomesticBorrowingCollection_df.find({},{"_id":0})))



# Streamlit app title
st.title("ECONOINSIGHTS")


#--------------------visualizations--------------------------------------------#


# Create a Streamlit sidebar button
if st.sidebar.button("Visualization of Past Data"):
      fig1 = px.line(df_ForeignBorrowingCollection, x='Year ', y='Foreign Borrowing', labels={'Year': 'Year', 'Foreign Borrowing': 'Y-Axis Label'})
      fig = px.line(df_MoneyPrintingCollection, x='Year ', y='Money_Printing', labels={'Year': 'Year', 'Money Printing': 'Y-Axis Label'})
      st.markdown("### Money Printing past data visualization")
      st.plotly_chart(fig)
      st.markdown("### Foreign Borrowing past data visualization")
      st.plotly_chart(fig1)
      fig3 = px.line(df_TaxationCollection, x='Year', y='Tax Revenue', labels={'Year': 'Year', 'Tax Revenue': 'Y-Axis Label'})
      st.markdown("### Tax Revenue past data visualization")
      st.plotly_chart(fig3)
      fig4 = px.line(df_DomesticBorrowingCollection, x='Year ', y='Domestic  Borrowing', labels={'Year': 'Year', 'Domestic Borrowing': 'Y-Axis Label'})
      st.markdown("### Domestic Borrowing past data visualization")
      st.plotly_chart(fig4)



#------------------------------------------------------------------------------#




# First Section
st.header("FORECAST")

# Date input for in order to make the prediction
pred_date = st.text_input("Select a Year for Prediction", value=2018)

input_data = pred_date
user_input_year_tax = int(pred_date)
user_input_year_dom = int(pred_date)
user_input_year_mon = int(pred_date)
user_input_year_debt = int(pred_date)



# Date input for YEAR (for TensorFlow model)
#input_data = st.sidebar.text_input("Select a Year for Foreign Borrowing Prediction")

# Text input for YEAR (for taxation model)
#user_input_year_tax = int(st.sidebar.number_input("Enter the Year for Taxation Forecasting"))


# Text input for YEAR (for domestic borrowings model)
#user_input_year_dom = int(st.sidebar.number_input("Enter the Year for Domestic Borrowings Forecasting"))

# Text input for YEAR (for Money printing model)
#user_input_year_mon = int(st.sidebar.number_input("Enter the Year for Money printing Forecasting"))

st.write("NOTE THAT THE AMOUNTS ARE IN MILLIONS")

# Define custom CSS styles for the card
card_style_1 = """
         background-color: #609944;
         padding: px;
         border-radius: 4px;
         box-shadow: 0 2px 2px 0 rgba(0, 0, 0, 0.2);
         margin-bottom: px;
         color: #FFFFFF;
"""

def separate_number(number):
    # Split the number into its integer and decimal parts
    integer_part, decimal_part = str(number).split('.')

    # Format the integer part with commas
    num_str = '{:,}'.format(int(integer_part))

    # Limit the decimal part to two decimal places
    decimal_part = decimal_part[:2]

    # Combine the formatted integer part and the limited decimal part
    result = f"Rs.{num_str}.{decimal_part}"

    return result

year_input  = user_input_year_tax


if year_input < 2022:
    # Access your collection
    collection = db["MoneyPrintingCollection"]

    # Convert the year input to an integer (if it's not already)
    year = int(year_input)

    # Define a query to find documents for the given year
    query = {"Year ":year}

    results = collection.find(query)
    money_pred = []
    for result in results:
        # Access fields within the document
        value1 = result["Money_Printing"]


        # Do something with the values
    money_pred =  value1
    money_pred_d1 = separate_number(money_pred)
    st.markdown(
    f'<div style="{card_style_1}">'
    f'<h5>Actual Money Printing amount for {year_input}: </h5>'
    f'<center><p><h3>{money_pred_d1}</h3></center></p>'
    '</div>',
    unsafe_allow_html=True
    )


    # Access your collection
    collection = db["ForeignBorrowingCollection"]

    # Convert the year input to an integer (if it's not already)
    year = int(year_input)

    # Define a query to find documents for the given year
    query = {"Year ":year}

    # Execute the query and retrieve the results
    #df = pd.DataFrame(list(collection.find(year)))
    #df
    #df = pd.DataFrame(collection.find(query))
    #df
    # Iterate through the results and do something with the data
    results = collection.find(query)
    predicted_debt = []
    for result in results:
        # Access fields within the document
        value1 = result["Foreign Borrowing"]
        #value2 = result["field2"]

        # Do something with the values
    predicted_debt = value1
    predicted_debt_d1 = separate_number(predicted_debt)
    st.markdown(
    f'<div style="{card_style_1}">'
    f'<h5>Actual Foreign Borrowing amount for {year_input}: </h5>'
    f'<center><p><h3>{predicted_debt_d1}</h3></center></p>'
    '</div>',
    unsafe_allow_html=True
    )



    # Access your collection
    collection = db["TaxationCollection"]

    # Convert the year input to an integer (if it's not already)
    year = int(year_input)

    # Define a query to find documents for the given year
    query = {"Year":year}

    # Execute the query and retrieve the results
    #df = pd.DataFrame(list(collection.find(year)))
    #df
    #df = pd.DataFrame(collection.find(query))
    #df
    # Iterate through the results and do something with the data
    results_t = collection.find(query)
    for result in results_t:
        # Access fields within the document
        value1 = result["Tax Revenue"]

        # Do something with the values
    tax_results = value1
    tax_results_d1 = separate_number(tax_results)
    st.markdown(
    f'<div style="{card_style_1}">'
    f'<h5>Actual Tax Revenue amount for {year_input}: </h5>'
    f'<center><p><h3>{tax_results_d1}</h3></center></p>'
    '</div>',
    unsafe_allow_html=True
    )


    collection = db["DomesticBorrowingCollection"]
    # Convert the year input to an integer (if it's not already)
    year = int(year_input)

    # Define a query to find documents for the given year
    query = {"Year ":year}

    # Execute the query and retrieve the results
    #df = pd.DataFrame(list(collection.find(year)))
    #df
    #df = pd.DataFrame(collection.find(query))
    #df
    # Iterate through the results and do something with the data
    results = collection.find(query)
    domestic_pred = []
    for result in results:
        # Access fields within the document
        value1 = result["Domestic  Borrowing"]
        #value2 = result["field2"]

        # Do something with the values
    domestic_pred =  value1
    domestic_pred_d1 = separate_number(domestic_pred)
    st.markdown(
    f'<div style="{card_style_1}">'
    f'<h5>Actual Domestic Borrowing amount for {year_input}: </h5>'
    f'<center><p><h3>{domestic_pred_d1}</h3></center></p>'
    '</div>',
    unsafe_allow_html=True
    )


    #government debt
    # Access your collection
    collection = db["GovernmentDebt"]

    # Convert the year input to an integer (if it's not already)
    year = int(year_input)

    # Define a query to find documents for the given year
    query = {"Year":year}

    # Execute the query and retrieve the results
    #df = pd.DataFrame(list(collection.find(year)))
    #df
    #df = pd.DataFrame(collection.find(query))
    #df
    # Iterate through the results and do something with the data
    results = collection.find(query)
    debt_pred = []
    for result in results:
        # Access fields within the document
        value1 = result["For_Borrowing"]
        value2 = result["Dom_Borrowing"]
        value3 = result["Tot_Debt"]

        # Do something with the values
    debt_pred = value3



    # Access your collection
    collection = db["GDPCollection"]
    # Convert the year input to an integer (if it's not already)
    year = int(year_input)

    # Define a query to find documents for the given year
    query = {"YEAR":year}

    # Execute the query and retrieve the results
    #df = pd.DataFrame(list(collection.find(year)))
    #df
    #df = pd.DataFrame(collection.find(query))
    #df
    # Iterate through the results and do something with the data
    results = collection.find(query)
    last_value_gdp = []
    for result in results:
        # Access fields within the document
        value1 = result["GDP-LKR"]


        # Do something with the values
    last_value_gdp = value1
else:
    # Predict Foreign Borrowing using TensorFlow model
    if input_data:
          given_year_tf = pd.to_datetime(input_data).timestamp()
          predicted_debt_tf = model_tf.predict(np.array([[given_year_tf]]))

          # Reshape the predicted debt to a 2D array
          predicted_debt_reshaped = predicted_debt_tf.reshape(-1, 1)

          # Load data from a pickle file
          with open('Foreign_model_scaler.pkl', 'rb') as file:
              minmax_scaler = pickle.load(file)

          # Inverse transform the predicted debt to get the original data
          original_predicted_debt = minmax_scaler.inverse_transform(predicted_debt_reshaped)

          predicted_value = original_predicted_debt[0][0]
          #st.write(f"Predicted Foreign Borrowing for {given_year_tf}: {predicted_value}")
          predicted_value_1 = separate_number(predicted_value)
          st.markdown(
            f'<div style="{card_style_1}">'
            f'<h5>Predicted Foreign Borrowing for {user_input_year_tax}: </h5>'
            f'<center><p><h3>{predicted_value_1}</h3></center></p>'
            '</div>',
            unsafe_allow_html=True
          )
          predicted_debt = predicted_value


    # Predict Taxation using the pickle model
    if user_input_year_tax:
        last_year_tax = 2021 - 2
        year_difference_tax = user_input_year_tax - last_year_tax
        periods_per_year_tax = 4
        forecast_horizon_tax = year_difference_tax * periods_per_year_tax
        prediction_tax = model_tax.forecast(forecast_horizon_tax)
        results_tax = prediction_tax.iloc[0]
        #st.write(f": ")
        results_tax_1 = separate_number(results_tax)
        st.markdown(
            f'<div style="{card_style_1}">'
            f'<h5>Predicted Taxation for {user_input_year_tax}</h5>'
            f'<center><p><h3>{results_tax_1}</h3></center></p>'
            '</div>',
            unsafe_allow_html=True
        )
        tax_results = results_tax


    # Predict Domestic Borrowings using the pickle model
    if user_input_year_dom:
        start_year_dom = 2021
        forecast_range_dom = range(start_year_dom, user_input_year_dom + 1)
        forecast_dom = model_dom.forecast(steps=len(forecast_range_dom))
        forecast_dom.index = forecast_range_dom
        #st.write(f"Predicted Domestic Borrowings for {user_input_year_dom}: {forecast_dom.iloc[-1]:.2f}")
        forecast_dom_f = forecast_dom.iloc[-1]
        results_dom = separate_number(forecast_dom_f)
        st.markdown(
            f'<div style="{card_style_1}">'
            f'<h5>Predicted Domestic Borrowing for {user_input_year_dom}: </h5>'
            f'<center><p><h3>{results_dom}</h3></center></p>'
            '</div>',
            unsafe_allow_html=True
        )
        domestic_pred = forecast_dom.iloc[-1]



    # Predict Money Printing using the pickle model
    if user_input_year_mon:
        start_year_mon = 2022
        forecast_range_mon = range(start_year_mon, user_input_year_mon + 1)
        forecast_mon = model_mon.forecast(steps=len(forecast_range_mon))
        forecast_mon.index = forecast_range_mon
        #st.write(f"Predicted money printing for {user_input_year_mon}: {forecast_mon.iloc[-1]:.2f}")
        forecast_dom_f = forecast_mon.iloc[-1]
        results_mon = separate_number(forecast_dom_f)
        st.markdown(
            f'<div style="{card_style_1}">'
            f'<h5>Predicted Money Printing for {user_input_year_mon}: </h5>'
            f'<center><p><h3>{results_mon}</h3></center></p>'
            '</div>',
            unsafe_allow_html=True
        )
        money_pred = forecast_mon.iloc[-1]


    # Predict Debt using thr pickle model
    if user_input_year_debt:
        forecast_steps = user_input_year_debt - 2021
        forecast = model_debt.get_forecast(steps=forecast_steps)
        forecasted_values = forecast.predicted_mean
        tot_gov_debt = forecasted_values[forecast_steps - 1]
        debt_pred = tot_gov_debt


    # Predict GDP using the pickle model
    if input_data:
        start_date = 2022
        date_range_gdp = pd.date_range(start_date, input_data,freq='y')
        forecast_gdp = model_gdp.forecast(steps=len(date_range_gdp))
        forecast_gdp.index = date_range_gdp
        last_value_gdp = forecast_gdp.iloc[-1]


#----------------start of simulation part--------------------------------------#

#Section header
st.header("OPTIMIZER")

#Define formulas---------------------------------------------------------------#

# Inflation
# m: "Money Printing"
# t: "Taxation"
# b: "borrowings"

def inf(m,t,d,f):
  values = [d, f, t, m]

  with open('svm_scaler.pkl', 'rb') as file:
      loaded_scaler = pickle.load(file)

  # Convert the list of values to a NumPy array
  array_2d = np.array(values)

  # Reshape the array to have one row and multiple columns
  array_2d = array_2d.reshape(1, -1)

  X_test_scaled = loaded_scaler.transform(array_2d)

  #Linear
  #return (6.42708112e-06 * d) - (2.35029896e-05 *f) + (7.91329095e-06* t) - (3.97426611e-07*m) + 7.663223081949728

  #SVR
  return (0.49161396 * X_test_scaled[0,0]) - (0.99352259 *X_test_scaled[0,1]) - (1.003063746* X_test_scaled[0,2]) - (0.65601281*X_test_scaled[0,3]) + 7.13781158
  #(0.49161396 * domestic) - (0.99352259 *foreign) - (1.003063746*taxation) - (0.65601281*money) + 7.13781158

  #Ridge
  #return (0.59462099 * X_test_scaled[0,0]) - (2.57325172 * X_test_scaled[0,1]) - (1.29945608 * X_test_scaled[0,2]) - (1.8897621 * X_test_scaled[0,3]) + 7.35055822395


# Debt Sustainablity
# gov_debt: "Total Government Debt as for now"
# foreign_debt: "Predicted Foreign Debt as for the 5th year"
# domestc_debt: "Predicted Domestic Debt as for the 5th year"
# money_printing: "Predicted Money Printing as for the 5th year"

#user input for GDP
#gdp_input = st.number_input("Please Enter Current GDP :", value=0, step=1)
gdp_input = last_value_gdp

def debt(current_total_gov_debt,foreign_debt,domestc_debt,money_printing):
  #Keeping GDP Fixed for the next five years
  #gov_debt current
  GDP = gdp_input
  return (current_total_gov_debt + foreign_debt + domestc_debt + money_printing ) /GDP




#Debt Sustainablity
gov_debt =  debt_pred

# User input - inflation_target
inflation_target = st.number_input("Please Enter Target Inflation Rate:  (eg : 3.0)", value=0.0)

# User input - debt_target
debt_target = st.number_input("Please Enter Target Debt-GDP ratio:  (eg : 5.0)", value=0.0)

# User input - expenditure amount
exp_amount = st.number_input("Please Enter Expected Government Expenditure: (Amount in millions) ", value=100000)



if pred_date is not None:
    #****   Intialize Values  *********

    #           Inflation
    # Consider min by 150%

    # Money Printing
    pred_m= money_pred
    m_min = pred_m/2
    m_max =(pred_m * 1.5)

    #Taxation
    pred_t= tax_results
    t_min = pred_t/2
    t_max = (pred_t *1.5)

    #Domestic Borrowings
    pred_b= domestic_pred
    b_min = pred_b/2
    b_max =(pred_b * 1.5)

    #Foreign Borrowings
    pred_f= predicted_debt
    f_min = pred_f/2
    f_max =(pred_f * 1.5)
else:
    st.write("Please Enter Year for Forecasting.")


#-----------------------------visualization------------------------------------#

# Calculate the total sum of the predicted values
#total_sum = predicted_debt + results + domestic_pred + money_pred

# Calculate the percentages for each value
#foreign_borrowing_percentage = (predicted_debt / total_sum) * 100
#taxation_percentage = (results / total_sum) * 100
#domestic_borrowings_percentage = (domestic_pred / total_sum) * 100
#money_printing_percentage = (money_pred / total_sum) * 100

# Data for the pie chart
#labels = ['Foreign Borrowing', 'Taxation', 'Domestic Borrowings', 'Money Printing']
#sizes = [foreign_borrowing_percentage, taxation_percentage, domestic_borrowings_percentage, money_printing_percentage]
#colors = ['#ff9999', '#66b3ff', '#99ff99', '#c2c2f0']

# Create the pie chart
#fig1, ax1 = plt.subplots()
#ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
#ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

# Display the pie chart using Streamlit
#st.pyplot(fig1)



#------------------------------------------------------------------------------#
# Define a button to trigger the Monte Carlo simulation
if st.button("Run Monte Carlo Simulation"):
    # Define the number of iterations
    n_iterations = 100000

    # Initializing empty list
    results = []

    # Run the Monte Carlo simulation
    for i in range(n_iterations):
        frames = []

        # Generate a random value for m, t, and b within the specified range

        # Inflation initialization
        mp = random.uniform(m_min, m_max)
        tax = random.uniform(t_min, t_max)
        f_bor = random.uniform(f_min, f_max)
        d_borrowings = random.uniform(b_min, b_max)

        # Debt initialization
        # for_debt = random.uniform(foreign_debt_min, foreign_debt_max)
        # dom_debt = random.uniform(domestic_debt_min, domestic_debt_max)
        # mp_debt = random.uniform(money_printing_min, money_printing_max)

        # Calculate the corresponding inflation value
        inf_val = inf(mp, tax, d_borrowings, f_bor)

        # Calculate the corresponding debt sustainability value
        debt_val = debt(gov_debt, f_bor, d_borrowings, mp)

        # Create a DataFrame with the calculated values and inputs
        df = pd.DataFrame({
            #'dom_inf': [dom_inf],
            #'for_inf': [for_inf],
            #'mp_inf': [mp_inf],
            'inf_val': [inf_val],
            'debt_val': [debt_val],
            'mp': [mp],
            'tax': [tax],
            'foreign': [f_bor],
            'dom': [d_borrowings]
        })

        # Append the DataFrame to the list of frames
        results.append(df)

    # Concatenate all the DataFrames in the list into a single DataFrame
    result_df = pd.concat(results, ignore_index=True)

    # Function to find optimized budgetary source values
    def optimize(df, inf_target, debt_target):
        filtered_df = df[
            (df['inf_val'] >= inf_target - 0.01) & (df['inf_val'] <= inf_target + 0.01) & (df['debt_val'] < debt_target)
        ]
        return filtered_df

    optimize_df = optimize(result_df, inflation_target, debt_target)

    final_df = optimize_df[
        (optimize_df['inf_val'] == optimize_df['inf_val'].min()) | (optimize_df['debt_val'] == optimize_df['debt_val'].min())]

    #st.dataframe(final_df)

    # Access 'mp', 'tax', 'foreign', 'dom' values for the second row (index 1)
    mp_value_2 = final_df.iloc[1]['mp']
    tax_value_2 = final_df.iloc[1]['tax']
    foreign_value_2 = final_df.iloc[1]['foreign']
    dom_value_2 = final_df.iloc[1]['dom']

    total_sum = foreign_value_2 + tax_value_2 + dom_value_2 + mp_value_2

    # Calculate the percentages for each value
    optimized_foreign_borrowing_percentage = (foreign_value_2 / total_sum) * 100
    optimized_taxation_percentage = (tax_value_2 / total_sum) * 100
    optimized_domestic_borrowings_percentage = (dom_value_2 / total_sum) * 100
    optimized_money_printing_percentage = (mp_value_2 / total_sum) * 100

    # Data for the pie chart
    labels = ['Foreign Borrowing', 'Taxation', 'Domestic Borrowings', 'Money Printing']
    sizes =[foreign_value_2, tax_value_2, dom_value_2, mp_value_2]
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#c2c2f0']

    # Create the pie chart
    fig5, ax2 = plt.subplots()
    ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax2.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    # Display the pie chart using Streamlit
    #header
    st.header("Optimal Percentages of Budgetary Sources")
    st.pyplot(fig5)


    st.header("Allocated Amounts (in millions):")


    mon_exp_amount = (optimized_money_printing_percentage/100) * exp_amount
    dom_exp_amount = (optimized_domestic_borrowings_percentage/100) * exp_amount
    tax_exp_amount = (optimized_taxation_percentage/100) * exp_amount
    for_exp_amount = (optimized_foreign_borrowing_percentage/100) * exp_amount


    # Define custom CSS styles for the card
    card_style = """
         background-color: #609944;
         padding: px;
         border-radius: 4px;
         box-shadow: 0 2px 2px 0 rgba(0, 0, 0, 0.2);
         margin-bottom: px;
         color: #FFFFFF;
    """

    def separate_number(number):
          # Split the number into its integer and decimal parts
          integer_part, decimal_part = str(number).split('.')

          # Format the integer part with commas
          num_str = '{:,}'.format(int(integer_part))

          # Limit the decimal part to two decimal places
          decimal_part = decimal_part[:2]

          # Combine the formatted integer part and the limited decimal part
          result = f"Rs.{num_str}.{decimal_part}"

          return result

    mon_exp_amount_1 = separate_number(mon_exp_amount)
    dom_exp_amount_1 = separate_number(dom_exp_amount)
    tax_exp_amount_1 = separate_number(tax_exp_amount)
    for_exp_amount_1 = separate_number(for_exp_amount)

    # Create card-like components for each result
    st.markdown(
        f'<div style="{card_style}">'
        f'<h5>Allocated Money Printing for given expenditure {exp_amount}:</h5>'
        f'<center><p><h3>{mon_exp_amount_1}</h3></center></p>'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div style="{card_style}">'
        f'<h5>Allocated Tax for given expenditure {exp_amount}:</h5>'
        f'<center><p><h3>{tax_exp_amount_1}</h3></center></p>'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div style="{card_style}">'
        f'<h5>Allocated Foreign Borrowing for given expenditure {exp_amount}:</h5>'
        f'<center><p><h3>{for_exp_amount_1}<h3></center></p>'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div style="{card_style}">'
        f'<h5>Allocated Domestic Borrowing for given expenditure {exp_amount}:</h5>'
        f'<center><p><h3>{dom_exp_amount_1}</h3></center></p>'
        '</div>',
        unsafe_allow_html=True
    )


