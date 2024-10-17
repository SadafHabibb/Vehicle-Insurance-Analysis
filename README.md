### Health Insurance Cross-Sell Dashboard - README

**Overview**  
This project is a web-based dashboard created using Streamlit. It aims to analyze and predict annual premiums for vehicle insurance based on customer data from a health insurance company looking to expand into the vehicle insurance sector.

The main goals are:
- To analyze potential customers for vehicle insurance.
- To predict an ideal selling price for vehicle insurance that maximizes revenue.

**Features**  
The dashboard includes three sections:
1. **Introduction**: 
   - Overview of the dataset and key variables.
   - Exploration of the data, including the ability to view a specified number of rows, and display statistical summaries.
   - Analysis of missing values and data completeness.

2. **Visualization**: 
   - Several data visualization options, such as:
     - **Pair Plot** using Seaborn for visualizing relationships between features.
     - **Bar Chart** to show the average annual premium for different customer categories.
     - **Line Chart** to observe trends in annual premiums across age groups.
     - **Pie Charts** to represent the distribution of responses based on vehicle damage and previous insurance history.

3. **Prediction**: 
   - Uses a linear regression model to predict the annual premium for vehicle insurance based on selected customer features.
   - Provides insights into model performance through metrics like Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE).
   - Displays a scatter plot comparing actual and predicted annual premiums.

**Technical Requirements**  
To run this dashboard, you need:
- Python 3.7 or above.
- Libraries: Streamlit, Pandas, NumPy, Seaborn, Matplotlib, Scikit-Learn, and PIL for image handling.

**Usage**  
- The dashboard can be run locally using Streamlit commands.
- The user can interact with the sidebar to switch between different sections (Introduction, Visualization, and Prediction) to analyze the data or run the prediction model.
- The dashboard dynamically updates visualizations and outputs based on user inputs.

**Dataset**  
The dataset (`train.csv`) includes features like customer age, gender, region, vehicle details, driving license status, and previous insurance history. These features help in understanding the potential for cross-selling vehicle insurance to health insurance customers. 

**Data Processing Steps**  
- Data cleaning includes handling missing values and transforming categorical variables (e.g., converting 'Yes'/'No' to 1/0).
- Visualizations and analysis focus on key variables that might influence insurance uptake, such as age, region, vehicle damage history, and insurance status.

**Model Evaluation**  
The linear regression model provides an initial approach to predicting insurance premiums. Evaluation metrics like MAE, MSE, and RMSE help assess the model's accuracy and guide further improvements.

**Potential Improvements**  
- Incorporating additional features or using more advanced machine learning models for better predictive accuracy.
- Enhancing data cleaning and feature engineering steps.
- Adding interactivity to visualizations for more dynamic data exploration.

This project demonstrates how data analysis and machine learning can support business decision-making in insurance cross-selling strategies.
