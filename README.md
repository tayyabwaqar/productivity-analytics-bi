# productivity-analytics-bi

# Garment Productivity Dashboard

## Overview

The Garment Productivity Dashboard is a Streamlit application designed to analyze and predict the productivity of employees in the garment manufacturing industry. This application provides insights into productivity trends, operational efficiency, and predictive analytics using machine learning models.

## Live Demo
https://ml-analytics-bi.streamlit.app/

## Features

- **Exploratory Data Analysis (EDA)**: Visualizations to understand the dataset, including:
  - Distribution of actual productivity
  - Correlation heatmap
  - Average productivity by department
  - Overtime vs. actual productivity scatter plot
- **Predictive Modeling**: Uses Random Forest and Decision Tree algorithms to predict productivity based on various factors.
- **Anomaly Detection**: Identifies and visualizes anomalies in productivity data.
- **Trend Analysis**: Displays productivity trends over time for different departments.
- **Departmental Analysis**: Compares productivity across different departments using box plots.
- **Customizable Reports**: Allows users to select metrics for generating downloadable reports.
- **User Feedback Mechanism**: Collects feedback from users directly within the app.
- **Data Warehousing Concepts**: Provides explanations of key data warehousing concepts.

## Technologies Used

- Python
- Streamlit
- Pandas
- Plotly
- Scikit-learn
- NumPy

## Installation

To run this application locally, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/garment-productivity-dashboard.git
   cd garment-productivity-dashboard

2. Create a virtual environment (optional but recommended):

    ```bash
    python -m venv venv
    source venv/bin/activate

3. Install the required packages:

   ```bash
    pip install -r requirements.txt

4. Ensure you have the dataset garment_productivity.csv in the same directory as the app.

## Usage

To run the Streamlit app, use the following command:
   ```bash
   streamlit run app.py


This will start a local server, and you can access the app in your web browser at http://localhost:8501.

## Contribution
Contributions are welcome! If you have suggestions for improvements or new features, please create an issue or submit a pull request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
Thanks to the contributors and the open-source community for their valuable resources and libraries that made this project possible.
