# NeuroStock: Stock Price Prediction System - Personal Finance

## Overview
NeuroStock is a sophisticated deep learning framework designed to predict stock prices with high accuracy, leveraging Long Short-Term Memory (LSTM), Convolutional Neural Network (CNN), and Hybrid CNN-LSTM models. Developed as a final semester project from June to December 2024 at Amity Institute of Information and Technology, Amity University Gurugram, NeuroStock processes historical financial data from the Yahoo Finance API (yfinance) to forecast stock prices over customizable horizons (7 to 60 days). Deployed as an interactive Streamlit web application, it allows users to select stock tickers, adjust model parameters, visualize predictions, and compare model performance in real time. Tested on Google (GOOG) stock data spanning 10 years, NeuroStock achieves robust performance, with the Hybrid CNN-LSTM model delivering balanced accuracy (RMSE: 9.2183, R²: 0.9134, Directional Accuracy: 0.4769, Directional Precision: 0.5056). The project bridges gaps in prior work by offering comprehensive model comparisons, directional trend evaluation, and a user-friendly interface, making it a valuable tool for investors, financial analysts, and researchers. 

## Objectives
NeuroStock aims to:
Accurate Forecasting: Develop deep learning models to predict stock prices with high numerical and directional accuracy for short- and long-term forecasts.
User Accessibility: Provide an intuitive Streamlit web platform for users to interact with complex AI models, visualize predictions, and derive actionable investment insights.
Scalability and Extensibility: Build a flexible framework supporting multiple stock tickers and future integration of additional data sources, such as sentiment analysis, to enhance predictive power.

## Methodology
NeuroStock follows a structured pipeline encompassing data collection, preprocessing, model development, training, evaluation, and web deployment.
### 1. Data Collection and Preprocessing
- Data Source: Historical stock data (Open, High, Low, Close, Volume) is retrieved via yfinance from Yahoo Finance for user-selected tickers (e.g., GOOG, AAPL) over a 10-year period.
- Preprocessing Steps:
- Null Handling: Missing values are dropped to ensure data integrity.
- Normalization: Closing prices are scaled to [0, 1] using MinMaxScaler from scikit-learn, with inverse transformation for visualization.
- Sequence Generation: A sliding window (default: 60 days, configurable 30–100 days) creates input sequences to predict the next day’s price.
- Dataset Splitting: Data is split into 70% training and 30% testing sets (adjustable 50–90% via the web interface).
These steps ensure compatibility with deep learning models and address preprocessing inconsistencies in prior studies.

### 2. Model Development and Training
NeuroStock implements three deep learning models using TensorFlow and Keras, each tailored to capture distinct patterns in financial time-series data:

### LSTM Model:
- Architecture: Two LSTM layers (50 units each, first returning sequences), Dropout (0.2), Dense layers (25, 1 units).
- Purpose: Captures long-term temporal dependencies in stock price trends.
- Configuration: Input shape (lookback, 1), Adam optimizer, Mean Squared Error (MSE) loss, MAE metric.
  
### CNN Model:
- Architecture: Two Conv1D layers (64, 32 filters, kernel_size=3, ReLU activation), MaxPooling1D (pool_size=2), Flatten, Dense layers (50, 1 units).
- Purpose: Extracts short-term local patterns and abrupt price changes.
- Configuration: Same optimizer and loss as LSTM.
  
### Hybrid CNN-LSTM Model:
- Architecture: Input layer, two Conv1D layers (64, 32 filters), MaxPooling1D, two LSTM layers (50 units), Dropout (0.2), Dense layers (25, 1 units).
- Purpose: Combines CNN’s feature extraction with LSTM’s sequential learning for balanced performance.
- Configuration: Uses Model API for custom architecture, same optimizer and loss.

Training Parameters:
  - Epochs: Up to 50 (configurable 10–100).
  - Batch Size: 32.
  - Validation Split: 10%.
  - EarlyStopping: Patience of 5 to prevent overfitting.
Users can adjust parameters (lookback, train-test split, epochs) via the Streamlit app, with live loss curves displayed.

### 3. Performance Evaluation
Models are evaluated using five metrics to provide a comprehensive assessment:

- Root Mean Squared Error (RMSE): Measures average squared prediction errors.
- Mean Absolute Error (MAE): Captures average error magnitude.
- R-squared (R²): Indicates variance explained by the model.
- Directional Accuracy: Percentage of correct up/down price movement predictions.
- Directional Precision: Proportion of correct positive (up) predictions among predicted positives.
  
Results are visualized in the Streamlit app with interactive plots and a metrics comparison table.

### 4. Web Deployment
NeuroStock is deployed as a Streamlit web application, offering a user-friendly interface for real-time analysis. Key features include:
- Ticker Selection: Choose from predefined tickers (e.g., AAPL, MSFT, GOOG) or enter custom tickers.
- Parameter Tuning: Adjust lookback (30–100 days), training split (50–90%), epochs (10–100), and prediction horizon (7–60 days).
- Visualization: Displays historical prices, 50/200-day moving averages, prediction vs. actual plots, error histograms, and training loss curves.
- Model Comparison: Interactive dashboard comparing RMSE, MAE, R², Accuracy, and Precision across models.
- Future Forecasting: Generates 7–60 day predictions with trend direction, volatility, price ranges, and investment insights (with disclaimers).
- Progress Tracking: Real-time progress bars and status updates during model training.

## Implementation
Technologies and Tools

- Programming Language: Python 3.8+
- Deep Learning Libraries: TensorFlow, Keras
- Data Processing: pandas, numpy, scikit-learn (MinMaxScaler)
- Data Retrieval: yfinance
- Visualization: matplotlib, seaborn
- Web Framework: Streamlit
- Development Environments: Jupyter Notebook, Visual Studio Code
- Hardware: Standard CPU/GPU configurations for model training

### Project Structure

neurostock/
├── app.py               # Streamlit web app
├── requirements.txt     # Project dependencies
├── models/              # Saved model weights (optional)
├── notebooks/           # Jupyter notebooks for experimentation
└── README.md            # This file

### Workflow
The Streamlit app (app.py) retrieves stock data via yfinance for user-selected tickers.
Data is preprocessed (normalized, sequenced) and split into training/testing sets based on user-defined parameters.
Three models (LSTM, CNN, Hybrid CNN-LSTM) are trained with live progress updates and loss curve visualization.
Predictions are generated, evaluated using five metrics, and visualized alongside historical trends.
Users can compare model performance, adjust settings, and generate future forecasts interactively.

## Results
Tested on GOOG stock with a 70:30 train-test split and 60-day lookback, NeuroStock delivers the following performance:

### LSTM Model:
- RMSE: 4.4134
- MAE: 3.4920
- R²: 0.9802
- Directional Accuracy: 0.4742
- Directional Precision: 0.5037
- Strength: Minimizes numerical errors, excels in capturing long-term trends.
- Weakness: Lower directional metrics, less effective for trend prediction.

### CNN Model:
- RMSE: 14.6011
- MAE: 10.6289
- R²: 0.7827
- Directional Accuracy: 0.5054
- Directional Precision: 0.5295
- Strength: Leads in directional metrics, ideal for short-term trend prediction.
- Weakness: Higher numerical errors, less precise for price forecasting.

### Hybrid CNN-LSTM Model:
- RMSE: 9.2183
- MAE: 6.9318
- R²: 0.9134
- Directional Accuracy: 0.4769
- Directional Precision: 0.5056
- Strength: Balances numerical accuracy and directional prediction, suitable for versatile applications.
- Visualizations: The Streamlit app displays:

Historical price plots with 50/200-day moving averages.
Prediction vs. actual plots for each model.
Error histograms showing error distribution.
Training loss curves for model convergence.
Future Forecasting: 30-day forecasts for GOOG indicate an upward trend, with metadata on volatility, price ranges, and peak days, enhancing investment decision-making.

## Challenges Overcome
Overfitting: Mitigated using Dropout layers (0.2) and EarlyStopping (patience=5) to ensure robust generalization.
- Data Quality: Addressed by dropping null values and normalizing data to maintain consistency.
- Model Complexity: Optimized architectures (e.g., 50 LSTM units, 64/32 CNN filters) for efficiency on standard hardware.
- User Accessibility: Streamlit interface simplifies complex model interactions, with progress bars and interactive visualizations.

## Limitations
- Data Scope: Relies on historical closing prices, excluding external factors like news sentiment or macroeconomic indicators.
- Market Anomalies: Performance may degrade during unpredictable events (e.g., black swan events).
- Uncertainty Quantification: Lacks confidence intervals for predictions, limiting risk assessment.
- Single Stock Focus: Primarily validated on GOOG; broader testing across stocks is needed.

## Contributions
NeuroStock advances stock market prediction research by:
- Comprehensive Model Comparison: Benchmarks LSTM, CNN, and Hybrid CNN-LSTM models, leveraging their complementary strengths.
- Directional Metrics: Incorporates Directional Accuracy and Precision, critical for investment decisions, addressing gaps in prior studies.
- Interactive Web App: Streamlit deployment makes advanced AI accessible to non-technical users, with real-time visualizations and parameter tuning.
- Scalable Framework: Supports multiple tickers and future enhancements, ensuring adaptability for diverse financial applications.

## Future Directions
To enhance NeuroStock’s capabilities, planned improvements include:
- Sentiment Analysis: Integrate Natural Language Processing (NLP) to analyze news and social media sentiment, improving responsiveness to market events.
- Multivariate Inputs: Incorporate technical indicators (e.g., RSI, MACD) and macroeconomic variables for robust modeling.
- Uncertainty Quantification: Implement Bayesian methods to provide prediction intervals for risk assessment.
- Cloud Deployment: Host on AWS or GCP with Docker for continuous learning and scalability.
- Cross-Market Validation: Test on diverse asset classes (e.g., ETFs, cryptocurrencies) and global exchanges.
- Reinforcement Learning: Develop trading strategies based on predictions to extend utility beyond forecasting.
