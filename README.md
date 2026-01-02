# COVID-19 Analytics Dashboard

**Programming for AI Course Project**  
**Developed by:** Arham Yunus Awan  
**Roll Number:** 2430-0007

---

## 📋 Project Overview

An interactive web-based analytics dashboard for exploring and analyzing COVID-19 pandemic data across US states. The project combines exploratory data analysis (EDA), advanced visualizations, and machine learning models to provide insights into the pandemic's progression and impact.

---

## ✨ Features

### 1. **Interactive Dashboard**
- Real-time KPIs: Total cases, deaths, hospitalizations, and mortality rates
- Multi-state comparison with customizable date ranges
- Dynamic data filtering and exploration

### 2. **Trend Analysis**
- Time-series visualization of COVID-19 metrics
- 7-day rolling averages for smoothed trend analysis
- Multiple chart types (line, area) for different perspectives

### 3. **Geographic Analysis**
- State-by-state comparison across multiple metrics
- Healthcare system impact visualization (ICU, ventilator usage)
- Top performers and hotspot identification

### 4. **Statistical Deep Dive**
- Correlation heatmaps between different COVID metrics
- Distribution analysis of key variables
- Comprehensive descriptive statistics

### 5. **Machine Learning Predictions**
- Random Forest Regressor for predicting daily new cases
- Customizable hyperparameters with real-time training
- Model performance metrics (R², RMSE, MAE, MSE)
- Feature importance visualization
- Model export functionality

### 6. **Data Export**
- Download filtered datasets in CSV format
- State-wise summary statistics
- Complete dataset access

---

## 🛠️ Technologies Used

- **Python 3.x**
- **Streamlit** - Interactive web application framework
- **Pandas & NumPy** - Data manipulation and analysis
- **Plotly** - Interactive visualizations
- **Scikit-learn** - Machine learning models
- **Matplotlib & Seaborn** - Statistical visualizations

---

## 📊 Dataset

**Source:** The COVID Tracking Project  
**Coverage:** US state-level COVID-19 data  
**Metrics:** 56+ variables including cases, deaths, testing, and hospitalization data  
**Records:** 20,780+ daily state reports

---

## 🚀 Getting Started

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd pai_project
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the dashboard**
   ```bash
   streamlit run app.py
   ```

4. **Run EDA script** (optional)
   ```bash
   python pai_project_eda_arham_yunus_awan_2430_0007.py
   ```

The dashboard will open automatically in your default web browser at `http://localhost:8501`

---

## 🎯 Key Insights

The dashboard enables users to:
- Track pandemic progression across different US states
- Identify trends and patterns in case numbers, deaths, and hospitalizations
- Compare healthcare system impacts across regions
- Predict future case trends using machine learning
- Make data-driven decisions based on statistical analysis

---

## 🤖 Machine Learning Models

### Random Forest Regressor
- **Purpose:** Predict daily new COVID-19 cases
- **Features:** Total cases, deaths, hospitalizations, ICU patients, ventilator usage, previous day cases
- **Evaluation Metrics:** R² Score, RMSE, MAE, MSE
- **Customization:** Adjustable hyperparameters (n_estimators, max_depth, min_samples_split)

---

## 📝 Usage Tips

1. **Select States:** Use the sidebar to choose states for comparison
2. **Adjust Date Range:** Filter data by specific time periods
3. **Quick Actions:** Use "Top 5" to view most affected states or "Reset" for default selection
4. **Explore Tabs:** Navigate through different analysis sections
5. **Train Models:** Customize ML hyperparameters and train models in real-time
6. **Export Data:** Download filtered or complete datasets for offline analysis

---

## 🎓 Course Information

This project was developed as part of the **Programming for AI** course, demonstrating:
- Data preprocessing and cleaning techniques
- Exploratory data analysis methodologies
- Interactive dashboard development
- Machine learning model implementation
- Data visualization best practices

---

## 📄 License

This project is created for educational purposes as part of the Programming for AI course.

---

## 🙏 Acknowledgments

- **Data Source:** The COVID Tracking Project
- **Course:** Programming for AI
- **Institution:** Sir Syed CASE Institute of Technology

---

**Last Updated:** January 2026