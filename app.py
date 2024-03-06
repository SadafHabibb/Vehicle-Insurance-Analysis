import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


st.sidebar.header("Dashboard")
st.sidebar.markdown("---")
app_mode = st.sidebar.selectbox('Select Page',['Introduction','Visualization','Prediction'])

df = pd.read_csv("train.csv")


if app_mode == "Introduction":

  st.image("veh.jpeg", use_column_width=True)
  st.title("Introduction")
  st.markdown("#### What is the relation between Health and Vehicle insurance?")

  st.markdown("##### Objectives")
  st.markdown("We aim to see what factors contributes to the purchase of Vehicle insurance.")

  st.markdown("Welcome to our Health Insurance Cross-Sell Dashboard!")
  st.markdown("Did you know? Globally, approximately 55% of the population lacks access to essential health services.")
  st.markdown("Furthermore, only 18% of the world's population has access to social security benefits that include health insurance coverage.")
  st.markdown("Our dashboard aims to bridge this gap by providing personalized recommendations and insights to help individuals make informed decisions about their insurance needs.")

  num = st.number_input('No. of Rows', 5, 10)

  head = st.radio('View from top (head) or bottom (tail)', ('Head', 'Tail'))
  if head == 'Head':
    st.dataframe(df.head(num))
  else:
    st.dataframe(df.tail(num))

  st.text('(Rows,Columns)')
  st.write(df.shape)

  st.markdown("##### Key Variables")
  st.markdown("- Gender of the customer")
  st.markdown("- Age of the customer")
  st.markdown("- Does the customer possess a Driving License")
  st.markdown("- Region of the customer")
  st.markdown("- Does the customer possess a Health insurance")
  st.markdown("- Age of the Vehicle")
  st.markdown("- Did the customer damage vehicle in past")
  st.markdown("- How much does customer pay for premium (INR)")
  st.markdown("- How long has the customer been associated with the company")

  st.markdown("From all these variables we wim to predict a price that the customers would be willing to pay for Vehicle Insurance.")
  st.markdown("Analysing the relationships between such as 'Vehicle Damage' and 'Previously_insured' with 'Response' will help us define our target audience.")
  st.markdown("Analysing relationships between 'Region' and 'Age' with 'Price' will help us define a price point.")

  st.markdown("### Description of Data")
  st.dataframe(df.describe())
  st.markdown("Descriptions for all quantitative data **(rank and streams)** by:")

  st.markdown("Count")
  st.markdown("Mean")
  st.markdown("Standard Deviation")
  st.markdown("Minimum")
  st.markdown("Quartiles")
  st.markdown("Maximum")

  st.markdown("### Missing Values")
  st.markdown("Null or NaN values.")

  dfnull = df.isnull().sum()/len(df)*100
  totalmiss = dfnull.sum().round(2)
  st.write("Percentage of total missing values:",totalmiss)
  st.write(dfnull)
  if totalmiss <= 30:
    st.success("We have less then 30 percent of missing values, which is good. This provides us with more accurate data as the null values will not significantly affect the outcomes of our conclusions. And no bias will steer towards misleading results. ")
  else:
    st.warning("Poor data quality due to greater than 30 percent of missing value.")
    st.markdown(" > Theoretically, 25 to 30 percent is the maximum missing values are allowed, there's no hard and fast rule to decide this threshold. It can vary from problem to problem.")

  st.markdown("### Completeness")
  st.markdown(" The ratio of non-missing values to total records in dataset and how comprehensive the data is.")

  st.write("Total data length:", len(df))
  nonmissing = (df.notnull().sum().round(2))
  completeness= round(sum(nonmissing)/len(df),2)

  st.write("Completeness ratio:",completeness)
  st.write(nonmissing)
  if completeness >= 0.80:
    st.success("We have completeness ratio greater than 0.85, which is good. It shows that the vast majority of the data is available for us to use and analyze. ")
  else:
    st.success("Poor data quality due to low completeness ratio( less than 0.85).")

elif app_mode == "Visualization":
  st.title("Visualization")

  # DATA TRIMMING
  # Changing "Yes" and "No" to 1 and 0
  df.loc[df['Vehicle_Damage'] == "Yes", 'Vehicle_Damage'] = 1
  df.loc[df['Vehicle_Damage'] == "No", 'Vehicle_Damage'] = 0

  # Deleting "Policy_Sales_Channel" column
  del df['Policy_Sales_Channel']

  # DATA VISUALISATION

  tab1, tab2, tab3, tab4 = st.tabs(["SNS Plot", "Bar Chart", "Line Chart", "Pie Plot"])

  #SNS plot
  tab1.subheader("SNS plot")
  sampled_df = df.sample(n=1000)
  fig = sns.pairplot(sampled_df)
  tab1.pyplot(fig)

  #Bar Graph
  # User input for x-variable
  columns = ['Region_Code', 'Gender', 'Vehicle_Age']
  x_variable = tab2.selectbox("Select x-variable:", columns)
  tab2.subheader(f"{x_variable} vs Price (INR)")
  data_by_variable = df.groupby(x_variable)['Annual_Premium'].mean()
  tab2.bar_chart(data_by_variable)

  #Line Graph
  tab3.subheader("Age vs Price")
  age_by_price = df.groupby('Age')['Annual_Premium'].mean()
  tab3.line_chart(age_by_price)

  #Pie Plot
  tab4.subheader("Response distribution by Vehicle Damage")
  response_counts = df.groupby(['Vehicle_Damage', 'Response']).size().unstack(fill_value=0)
  fig, ax = plt.subplots()
  colors = ['#ff9999','#66b3ff']
  damage_counts = response_counts.loc[1]
  percentages = (damage_counts.values / damage_counts.sum()) * 100
  labels = ['Yes', 'No']
  ax.pie(percentages, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
  ax.axis('equal')
  tab4.pyplot(fig)

  #Pie Plot2
  tab4.subheader("Response Distribution by Not Previously Insured")
  response_counts = df.groupby(['Previously_Insured', 'Response']).size().unstack(fill_value=0)
  fig, ax = plt.subplots()
  colors = ['#ff9999','#66b3ff']
  prev_insurance_counts = response_counts.loc[0]
  percentages = (prev_insurance_counts.values / prev_insurance_counts.sum()) * 100
  labels = ['Yes', 'No']
  ax.pie(percentages, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
  ax.axis('equal')
  tab4.pyplot(fig)



elif app_mode == "Prediction":
  # Changing "Yes" and "No" to 1 and 0
  df.loc[df['Vehicle_Damage'] == "Yes", 'Vehicle_Damage'] = 1
  df.loc[df['Vehicle_Damage'] == "No", 'Vehicle_Damage'] = 0
  st.title("Prediction")
  X = df[['Age', 'Region_Code', 'Driving_License','Vehicle_Damage', 'Previously_Insured']]
  y = df['Annual_Premium']
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  lin_reg = LinearRegression()
  lin_reg.fit(X_train,y_train)
  pred = lin_reg.predict(X_test)

  plt.figure(figsize=(10,7))
  plt.title("Actual vs. predicted Annual Premiums",fontsize=25)
  plt.xlabel("Actual test set Annual Premiums",fontsize=18)
  plt.ylabel("Predicted Annual Premiums", fontsize=18)
  plt.scatter(x=y_test,y=pred)
  plt.savefig('prediction.png')
  st.image('prediction.png')

  # Model Evaluation
  st.markdown("Evaluation")
  coeff_df = pd.DataFrame(lin_reg.coef_, X.columns, columns=['Coefficient'])
  st.dataframe(coeff_df)
  MAE = metrics.mean_absolute_error(y_test, pred)
  MSE = metrics.mean_squared_error(y_test, pred)
  RMSE = np.sqrt(metrics.mean_squared_error(y_test, pred))
  st.write('MAE:', MAE)
  st.write('MSE:', MSE)
  st.write('RMSE:', RMSE)



