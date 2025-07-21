#  Predictive Analysis of Market Trends

A Streamlit web application for analyzing and predicting **Inflation Rates** using traditional Machine Learning models (Random Forest, SVM, XGBoost) and Deep Learning (LSTM). Upload your dataset, visualize feature correlations, train models, and download predictions.

---

##  Features

-  Upload custom economic data in CSV format
-  Visualize correlations using a heatmap
-  Train models:  
  - Random Forest  
  - Support Vector Machine (SVM)  
  - XGBoost  
  - LSTM (Deep Learning)
-  Compare model performance (MAE, MSE, R²)
-  Download predictions as CSV
-  Side-by-side model comparisons

---

##  Dataset Format

Make sure your CSV dataset includes the following:

- `observation_date` — column with year/month (e.g., `2001`, `2020-03`, etc.)
- Other columns:  
  e.g., `inflation_rate`, `GDP`, `CPI`, `unemployment_rate`, `interest_rate`, etc.

Example CSV structure:

```csv
observation_date,inflation_rate,GDP,CPI,unemployment_rate
2000,3.2,1000,200,5.1
2001,2.9,1050,205,5.3
```
## Project Stages
### 1.  Data Collection
 -Economic data was collected from the FRED API, including macroeconomic indicators like inflation rate, GDP, CPI, unemployment rate, and interest rate.

### 2.  Data Imputation (Daily Frequency)
 -Missing data points were imputed, and the dataset was resampled to a daily frequency to ensure temporal consistency for time-series modeling.

### 3.  Hourly Conversion using Fourier Transform
 -A Fourier Transform-based method was used to convert daily data into hourly granularity, helping models—especially LSTM—capture finer temporal patterns.

### 4.  Feature Engineering & Visualization
 -Created correlation heatmaps, distribution plots, and time series charts to understand inter-feature relationships and improve model inputs. Key economic variables were selected based on correlation strength and significance.

### 5.  Model Exploration
 -Multiple models were trained and evaluated:

🟢 Random Forest

🔵 Support Vector Machine (SVM)

🟡 XGBoost

🔴 LSTM (Deep Learning)

Each model’s performance was compared using metrics like MAE, MSE, and R² Score.

### 6. 🌐 Streamlit Integration

- The final solution was integrated into a Streamlit web app, allowing users to:

- Upload custom datasets

- Visualize correlations

- Train and evaluate models

- Download prediction results in CSV format

## Installation process:
- clone the repository:
<br> open your teminal and enter
```
git clone https://github.com/your-username/predictive-analysis-market-trends.git
cd predictive-analysis-market-trends
```
- creating virtual environment
```
  # For Windows:
python -m venv venv
venv\Scripts\activate

# For macOS/Linux:
python3 -m venv venv
source venv/bin/activate
```
- Installing dependencies
```
pip install -r requirements.txt
```
- Running the app
```
streamlit run updated_app.py
```
##  Contact

For questions, feedback, or contributions, feel free to reach out:

**Shashwat Thakur**  
Email: [shashwatthakur16129@gmail.com](mailto:shashwatthakur16129@gmail.com)



