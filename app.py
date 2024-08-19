# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 13:21:55 2024

@author: Tayyab
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import io

# Set the title and layout of the app
st.set_page_config(page_title="Garment Productivity Dashboard", layout="wide")

st.title('Garment Productivity Dashboard')
st.markdown("""
Analyze and predict the productivity of garment employees using various factors.
""")

# Load the dataset
@st.cache_data
def load_data():
    data = pd.read_csv('garment_productivity.csv')
    data['date'] = pd.to_datetime(data['date'], format='%m/%d/%Y')  # Ensure date is in datetime format
    return data

data = load_data()

# Display the dataset
st.subheader('Dataset Overview')
st.write(data.head())

# EDA Section
st.header('Exploratory Data Analysis (EDA)')

# Distribution of Actual Productivity
st.subheader('Distribution of Actual Productivity')
st.markdown("This histogram shows the distribution of actual productivity values, helping identify the overall performance trends.")
hist_fig = px.histogram(data, x='actual_productivity', nbins=20, title='Distribution of Actual Productivity')
st.plotly_chart(hist_fig, use_container_width=True)

# Correlation Heatmap
st.subheader('Correlation Heatmap')
st.markdown("This heatmap shows the correlation between different features in the dataset. Strong correlations can indicate relationships worth exploring.")
# Select only numeric columns for correlation
correlation_matrix = data.select_dtypes(include=[np.number]).corr()
heatmap_fig = px.imshow(correlation_matrix, text_auto=True, title='Correlation Heatmap')
st.plotly_chart(heatmap_fig, use_container_width=True)

# Productivity by Department
st.subheader('Average Productivity by Department')
st.markdown("This bar chart displays the average productivity for each department, allowing for quick comparisons.")
avg_prod_by_dept = data.groupby('department')['actual_productivity'].mean().reset_index()
bar_fig = px.bar(avg_prod_by_dept, x='department', y='actual_productivity', title='Average Productivity by Department')
st.plotly_chart(bar_fig, use_container_width=True)

# Overtime vs. Productivity
st.subheader('Overtime vs. Actual Productivity')
st.markdown("This scatter plot examines the relationship between overtime and actual productivity, helping identify trends.")
scatter_fig = px.scatter(data, x='over_time', y='actual_productivity', title='Overtime vs. Actual Productivity', trendline='ols')
st.plotly_chart(scatter_fig, use_container_width=True)

# Performance Metrics
st.subheader('Performance Metrics')
kpis = data[['actual_productivity', 'over_time', 'incentive', 'idle_time']].mean()
industry_benchmark = 0.75  # Example benchmark value
st.write(f"**Average Actual Productivity:** {kpis['actual_productivity']:.2f} (Benchmark: {industry_benchmark})")
st.write(f"**Average Overtime (minutes):** {kpis['over_time']:.2f}")
st.write(f"**Average Incentive (BDT):** {kpis['incentive']:.2f}")
st.write(f"**Average Idle Time (minutes):** {kpis['idle_time']:.2f}")

# Trend Analysis
st.subheader('Trend Analysis')
st.markdown("This graph shows the productivity trends over time for different departments. It helps identify seasonal patterns and productivity fluctuations.")
trend_fig = px.line(data, x='date', y='actual_productivity', color='department', title='Productivity Trends Over Time')
st.plotly_chart(trend_fig, use_container_width=True)

# Departmental Analysis
st.subheader('Departmental Analysis')
st.markdown("This box plot compares productivity across departments, highlighting variations and identifying which departments consistently perform better or worse.")
dept_fig = px.box(data, x='department', y='actual_productivity', title='Productivity by Department')
st.plotly_chart(dept_fig, use_container_width=True)

# Target vs. Actual Productivity
st.subheader('Target vs. Actual Productivity')
st.markdown("This scatter plot compares targeted productivity with actual productivity, showing how well teams meet their goals.")
target_fig = px.scatter(data, x='targeted_productivity', y='actual_productivity', color='department', title='Target vs. Actual Productivity')
st.plotly_chart(target_fig, use_container_width=True)

# Operational Efficiency
st.subheader('Operational Efficiency')
st.markdown("This scatter plot examines the relationship between style changes and idle time, helping identify bottlenecks in production processes.")
efficiency_fig = px.scatter(data, x='no_of_style_change', y='idle_time', size='no_of_workers', color='department', title='Idle Time vs. Style Changes')
st.plotly_chart(efficiency_fig, use_container_width=True)

# Anomaly Detection
def detect_anomalies(data, threshold=2):
    mean = np.mean(data)
    std_dev = np.std(data)
    anomalies = data[(data > mean + threshold * std_dev) | (data < mean - threshold * std_dev)]
    return anomalies

anomalies = detect_anomalies(data['actual_productivity'])

# Anomaly Visualization
st.subheader("Anomalies in Productivity")
st.markdown("This scatter plot highlights anomalies in productivity, helping identify unusual performance deviations.")
anomaly_fig = px.scatter(data, x='date', y='actual_productivity', title='Anomalies in Productivity')
anomaly_fig.add_trace(go.Scatter(x=data['date'], y=anomalies, mode='markers', marker=dict(color='red', size=10), name='Anomalies'))
st.plotly_chart(anomaly_fig, use_container_width=True)

# Predictive Modeling with Random Forest
st.subheader('Predictive Modeling')
features = ['no_of_workers', 'over_time', 'incentive', 'idle_time', 'no_of_style_change']
X = data[features]
y = data['actual_productivity']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

rf_mse = mean_squared_error(y_test, rf_predictions)
st.write(f"**Mean Squared Error of Random Forest Prediction:** {rf_mse:.4f}")

# Decision Tree Model
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)
dt_predictions = dt_model.predict(X_test)

dt_mse = mean_squared_error(y_test, dt_predictions)
st.write(f"**Mean Squared Error of Decision Tree Prediction:** {dt_mse:.4f}")

# Scenario Analysis
st.subheader('Scenario Analysis')
worker_count = st.slider('Number of Workers', min_value=int(data['no_of_workers'].min()), max_value=int(data['no_of_workers'].max()), value=int(data['no_of_workers'].mean()))
overtime = st.slider('Overtime (minutes)', min_value=int(data['over_time'].min()), max_value=int(data['over_time'].max()), value=int(data['over_time'].mean()))
incentive = st.slider('Incentive (BDT)', min_value=int(data['incentive'].min()), max_value=int(data['incentive'].max()), value=int(data['incentive'].mean()))
idle_time = st.slider('Idle Time (minutes)', min_value=int(data['idle_time'].min()), max_value=int(data['idle_time'].max()), value=int(data['idle_time'].mean()))
style_changes = st.slider('Number of Style Changes', min_value=int(data['no_of_style_change'].min()), max_value=int(data['no_of_style_change'].max()), value=int(data['no_of_style_change'].mean()))

# Ensure the scenario input uses the same features as the model
scenario_input = pd.DataFrame([[worker_count, overtime, incentive, idle_time, style_changes]], columns=features)
rf_scenario_prediction = rf_model.predict(scenario_input)[0]
dt_scenario_prediction = dt_model.predict(scenario_input)[0]

st.write(f"**Predicted Productivity for Scenario (Random Forest):** {rf_scenario_prediction:.4f}")
st.write(f"**Predicted Productivity for Scenario (Decision Tree):** {dt_scenario_prediction:.4f}")

# Visualize Predicted Productivity with a Speedometer
st.subheader('Predicted Productivity Visualization')
gauge_fig = go.Figure(go.Indicator(
    mode="gauge",
    value=rf_scenario_prediction * 100,  # Convert to percentage
    domain={'x': [0, 1], 'y': [0, 1]},
    title={'text': "Predicted Productivity (%)"},
    gauge={
        'axis': {'range': [-100, 100]},
        'bar': {'color': "darkblue"},
        'steps': [
            {'range': [-100, 0], 'color': "red"},
            {'range': [0, 50], 'color': "yellow"},
            {'range': [50, 100], 'color': "green"}
        ],
    }
))

# Add annotation to center the predicted value
gauge_fig.add_annotation(
    x=0.5,
    y=0.35,
    text=f"{rf_scenario_prediction * 100:.2f}%",  # Display predicted productivity
    showarrow=False,
    font=dict(size=35)
)

st.plotly_chart(gauge_fig, use_container_width=True)

# Customizable Reports
st.sidebar.header("Report Customization")
selected_metrics = st.sidebar.multiselect(
    "Select Metrics for Report",
    options=['actual_productivity', 'over_time', 'incentive', 'idle_time'],
    default=['actual_productivity', 'over_time']
)

if st.sidebar.button("Generate Report"):
    report_data = data[selected_metrics]
    buffer = io.StringIO()
    report_data.to_csv(buffer)
    buffer.seek(0)
    st.sidebar.download_button(
        label="Download Report",
        data=buffer,
        file_name="custom_report.csv",
        mime="text/csv"
    )

# User Feedback
st.sidebar.header("Feedback")
with st.sidebar.form("feedback_form"):
    feedback = st.text_area("Please provide your feedback or suggestions:")
    submitted = st.form_submit_button("Submit")
    if submitted:
        st.sidebar.success("Thank you for your feedback!")

# Data Warehousing Concepts
st.sidebar.header("Data Concepts")
st.sidebar.markdown("""
**Data Warehousing Concepts:**

- **Data Mart**: A subset of a data warehouse focused on a specific business line.
- **Data Lake**: A centralized repository for storing raw data in its native format.
- **ETL Process**: Extract, Transform, Load - the process of moving data from source systems to a data warehouse.
""")

# Add user guidance
st.info("Use the sliders to simulate different scenarios and see the predicted productivity.")