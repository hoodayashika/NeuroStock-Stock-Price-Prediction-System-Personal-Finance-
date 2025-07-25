import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv1D, MaxPooling1D, Flatten, Input
from tensorflow.keras.callbacks import EarlyStopping

# Set page configuration
st.set_page_config(page_title="NeuroStock", layout="wide")

# App title
st.title("NeuroStock: Stock Market Forecasting")
st.markdown("---")

# Sidebar for model parameters
st.sidebar.header("Model Parameters")
lookback = st.sidebar.slider("Lookback Period (Days)", 30, 100, 60)
train_split = st.sidebar.slider("Training Data Split (%)", 50, 90, 70)
epochs = st.sidebar.slider("Training Epochs", 10, 100, 50)
future_days = st.sidebar.slider("Future Prediction Days", 7, 60, 30)

# Functions for model creation, training and prediction
def calculate_metrics(y_true, y_pred):
    """Calculate accuracy and precision for directional prediction"""
    # Convert to directional movements (1 for up, 0 for down)
    y_true_dir = np.array([1 if y_true[i] > y_true[i-1] else 0 for i in range(1, len(y_true))])
    y_pred_dir = np.array([1 if y_pred[i] > y_pred[i-1] else 0 for i in range(1, len(y_pred))])
    
    # Calculate metrics
    acc = accuracy_score(y_true_dir, y_pred_dir)
    prec = precision_score(y_true_dir, y_pred_dir, zero_division=0)
    
    return acc, prec

def create_sequences(data, lookback):
    """Create sequences for LSTM & CNN models"""
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

def build_lstm_model(lookback):
    """Build LSTM model"""
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(lookback, 1)),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=25),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

def build_cnn_model(lookback):
    """Build CNN model"""
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(lookback, 1)),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=32, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(50, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

def build_hybrid_model(lookback):
    """Build hybrid CNN-LSTM model"""
    input_layer = Input(shape=(lookback, 1))
    
    # CNN part
    cnn_part = Conv1D(filters=64, kernel_size=3, activation='relu')(input_layer)
    cnn_part = MaxPooling1D(pool_size=2)(cnn_part)
    cnn_part = Conv1D(filters=32, kernel_size=3, activation='relu')(cnn_part)
    
    # LSTM part
    lstm_part = LSTM(units=50, return_sequences=True)(cnn_part)
    lstm_part = Dropout(0.2)(lstm_part)
    lstm_part = LSTM(units=50, return_sequences=False)(lstm_part)
    lstm_part = Dropout(0.2)(lstm_part)
    
    # Dense layers
    dense_layer = Dense(units=25)(lstm_part)
    output_layer = Dense(units=1)(dense_layer)
    
    # Create model
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

def make_future_predictions(model, data, lookback, future_days, scaler):
    """Generate future predictions"""
    last_sequence = data[-lookback:].reshape(1, lookback, 1)
    future_predictions = []
    current_sequence = last_sequence.copy()
    
    for _ in range(future_days):
        # Get prediction for next day
        pred = model.predict(current_sequence, verbose=0)
        # Add prediction to list
        future_predictions.append(pred[0,0])
        # Update sequence by removing oldest value and adding the prediction
        pred_reshaped = np.array([[[pred[0,0]]]])
        current_sequence = np.append(current_sequence[:,1:,:], pred_reshaped, axis=1)
    
    # Inverse transform the predictions
    future_predictions = np.array(future_predictions).reshape(-1, 1)
    future_predictions = scaler.inverse_transform(future_predictions)
    
    return future_predictions

# Main app
def main():
    # Stock ticker selection
    available_tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'V', 'WMT']
    default_ticker = 'AAPL'
    
    col1, col2 = st.columns([3, 1])
    with col1:
        stock = st.selectbox("Select Stock Ticker", available_tickers, index=available_tickers.index(default_ticker))
    with col2:
        custom_ticker = st.text_input("Or Enter Custom Ticker")
        if custom_ticker:
            stock = custom_ticker.upper()
    
    # Set date range
    end = datetime.now()
    start = datetime(end.year - 10, end.month, end.day)  # 10 years of data
    
    # Load data button
    if st.button("Load Stock Data"):
        with st.spinner(f"Loading data for {stock}..."):
            try:
                # Download stock data
                stock_data = yf.download(stock, start, end)
                
                if len(stock_data) == 0:
                    st.error(f"No data found for ticker: {stock}. Please check the ticker symbol.")
                    return
                
                # Display data info
                st.subheader(f"{stock} Stock Data")
                # Add a container with fixed height and scrollbar for the dataframe
                st.write("Historical data from past 10 years:")
                st.dataframe(stock_data, height=300)
                
                # Plot closing price
                st.subheader("Historical Closing Price")
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(stock_data['Close'], label="Closing Price", color='blue')
                ax.set_xlabel("Date")
                ax.set_ylabel("Stock Price")
                ax.set_title(f"{stock} Closing Price")
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                
                # Compute moving averages
                stock_data['MA_50'] = stock_data['Close'].rolling(50).mean()
                stock_data['MA_200'] = stock_data['Close'].rolling(200).mean()
                
                # Plot closing price with moving averages
                st.subheader("Stock Price with Moving Averages")
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(stock_data['Close'], label="Closing Price", color='blue')
                ax.plot(stock_data['MA_50'], label="50-Day MA", color='green', linestyle='dashed')
                ax.plot(stock_data['MA_200'], label="200-Day MA", color='red', linestyle='dashed')
                ax.set_xlabel("Date")
                ax.set_ylabel("Stock Price")
                ax.set_title(f"{stock} Price with Moving Averages")
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                
                # Prepare data for modeling
                # Use only 'Close' prices for simplicity
                data = stock_data[['Close']].copy()
                data.dropna(inplace=True)
                
                # Normalize data
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled_data = scaler.fit_transform(data)
                
                # Create sequences
                X, y = create_sequences(scaled_data, lookback)
                
                # Reshape input to be [samples, time steps, features]
                X = np.reshape(X, (X.shape[0], X.shape[1], 1))
                
                # Split into train and test sets
                train_size = int(len(X) * (train_split / 100))
                X_train, X_test = X[:train_size], X[train_size:]
                y_train, y_test = y[:train_size], y[train_size:]
                
                # Display training info
                st.info(f"Training on {train_size} samples, Testing on {len(X_test)} samples")
                
                # Train models
                st.subheader("Model Training")
                
                # Create tabs for different models
                lstm_tab, cnn_tab, hybrid_tab, comparison_tab, future_tab = st.tabs([
                    "LSTM Model", "CNN Model", "Hybrid CNN-LSTM Model", 
                    "Models Comparison", "Future Prediction"
                ])
                
                early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
                
                # LSTM Model
                with lstm_tab:
                    st.write("Training LSTM Model...")
                    lstm_model = build_lstm_model(lookback)
                    
                    # Show model summary
                    st.code(lstm_model.summary())
                    
                    # Train LSTM model
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    class CustomCallback(tf.keras.callbacks.Callback):
                        def on_epoch_end(self, epoch, logs=None):
                            progress = (epoch + 1) / epochs
                            progress_bar.progress(progress)
                            status_text.text(f"Training LSTM Model: {epoch+1}/{epochs} epochs")
                    
                    history_lstm = lstm_model.fit(
                        X_train, y_train,
                        epochs=epochs,
                        batch_size=32,
                        validation_split=0.1,
                        callbacks=[early_stop, CustomCallback()],
                        verbose=0
                    )
                    
                    # Plot training history
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(history_lstm.history['loss'], label='Training Loss', color='blue')
                    ax.plot(history_lstm.history['val_loss'], label='Validation Loss', color='red')
                    ax.set_title('LSTM Model Loss')
                    ax.set_xlabel('Epochs')
                    ax.set_ylabel('Loss')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    
                    # Generate LSTM predictions
                    lstm_predictions = lstm_model.predict(X_test, verbose=0)
                
                # CNN Model
                with cnn_tab:
                    st.write("Training CNN Model...")
                    cnn_model = build_cnn_model(lookback)
                    
                    # Show model summary
                    st.code(cnn_model.summary())
                    
                    # Train CNN model
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    class CustomCallback(tf.keras.callbacks.Callback):
                        def on_epoch_end(self, epoch, logs=None):
                            progress = (epoch + 1) / epochs
                            progress_bar.progress(progress)
                            status_text.text(f"Training CNN Model: {epoch+1}/{epochs} epochs")
                    
                    history_cnn = cnn_model.fit(
                        X_train, y_train,
                        epochs=epochs,
                        batch_size=32,
                        validation_split=0.1,
                        callbacks=[early_stop, CustomCallback()],
                        verbose=0
                    )
                    
                    # Plot training history
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(history_cnn.history['loss'], label='Training Loss', color='blue')
                    ax.plot(history_cnn.history['val_loss'], label='Validation Loss', color='red')
                    ax.set_title('CNN Model Loss')
                    ax.set_xlabel('Epochs')
                    ax.set_ylabel('Loss')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    
                    # Generate CNN predictions
                    cnn_predictions = cnn_model.predict(X_test, verbose=0)
                
                # Hybrid CNN-LSTM Model
                with hybrid_tab:
                    st.write("Training Hybrid CNN-LSTM Model...")
                    hybrid_model = build_hybrid_model(lookback)
                    
                    # Show model summary
                    st.code(hybrid_model.summary())
                    
                    # Train hybrid model
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    class CustomCallback(tf.keras.callbacks.Callback):
                        def on_epoch_end(self, epoch, logs=None):
                            progress = (epoch + 1) / epochs
                            progress_bar.progress(progress)
                            status_text.text(f"Training Hybrid Model: {epoch+1}/{epochs} epochs")
                    
                    history_hybrid = hybrid_model.fit(
                        X_train, y_train,
                        epochs=epochs,
                        batch_size=32,
                        validation_split=0.1,
                        callbacks=[early_stop, CustomCallback()],
                        verbose=0
                    )
                    
                    # Plot training history
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(history_hybrid.history['loss'], label='Training Loss', color='blue')
                    ax.plot(history_hybrid.history['val_loss'], label='Validation Loss', color='red')
                    ax.set_title('Hybrid CNN-LSTM Model Loss')
                    ax.set_xlabel('Epochs')
                    ax.set_ylabel('Loss')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    
                    # Generate hybrid predictions
                    hybrid_predictions = hybrid_model.predict(X_test, verbose=0)
                
                # Models Comparison
                with comparison_tab:
                    st.write("Comparing Model Performances...")
                    
                    # Inverse transform predictions
                    lstm_predictions = scaler.inverse_transform(lstm_predictions)
                    cnn_predictions = scaler.inverse_transform(cnn_predictions)
                    hybrid_predictions = scaler.inverse_transform(hybrid_predictions)
                    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
                    
                    # Calculate metrics
                    # LSTM
                    lstm_rmse = np.sqrt(mean_squared_error(y_test_actual, lstm_predictions))
                    lstm_mae = mean_absolute_error(y_test_actual, lstm_predictions)
                    lstm_r2 = r2_score(y_test_actual, lstm_predictions)
                    lstm_acc, lstm_prec = calculate_metrics(y_test_actual.flatten(), lstm_predictions.flatten())
                    
                    # CNN
                    cnn_rmse = np.sqrt(mean_squared_error(y_test_actual, cnn_predictions))
                    cnn_mae = mean_absolute_error(y_test_actual, cnn_predictions)
                    cnn_r2 = r2_score(y_test_actual, cnn_predictions)
                    cnn_acc, cnn_prec = calculate_metrics(y_test_actual.flatten(), cnn_predictions.flatten())
                    
                    # Hybrid CNN-LSTM
                    hybrid_rmse = np.sqrt(mean_squared_error(y_test_actual, hybrid_predictions))
                    hybrid_mae = mean_absolute_error(y_test_actual, hybrid_predictions)
                    hybrid_r2 = r2_score(y_test_actual, hybrid_predictions)
                    hybrid_acc, hybrid_prec = calculate_metrics(y_test_actual.flatten(), hybrid_predictions.flatten())
                    
                    # Create a dataframe with all metrics for comparison
                    metrics_df = pd.DataFrame({
                        'Model': ['LSTM', 'CNN', 'Hybrid CNN-LSTM'],
                        'RMSE': [lstm_rmse, cnn_rmse, hybrid_rmse],
                        'MAE': [lstm_mae, cnn_mae, hybrid_mae],
                        'R²': [lstm_r2, cnn_r2, hybrid_r2],
                        'Accuracy': [lstm_acc, cnn_acc, hybrid_acc],
                        'Precision': [lstm_prec, cnn_prec, hybrid_prec]
                    })
                    
                    # Display metrics
                    st.subheader("Model Performance Metrics")
                    st.dataframe(metrics_df.style.highlight_min(subset=['RMSE', 'MAE'], color='lightgreen')
                                .highlight_max(subset=['R²', 'Accuracy', 'Precision'], color='lightgreen'))
                    
                    # Get dates for the test period
                    test_dates = data.index[train_size+lookback:].tolist()
                    
                    # Plot predictions vs actual
                    st.subheader("Predictions vs Actual")
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.plot(test_dates, y_test_actual, label='Actual Prices', color='black', linewidth=2)
                    ax.plot(test_dates, lstm_predictions, label='LSTM Predictions', color='blue', alpha=0.8)
                    ax.plot(test_dates, cnn_predictions, label='CNN Predictions', color='green', alpha=0.8)
                    ax.plot(test_dates, hybrid_predictions, label='Hybrid CNN-LSTM Predictions', color='red', alpha=0.8)
                    ax.set_title(f'Stock Price Prediction Comparison for {stock}', fontsize=14)
                    ax.set_xlabel('Date', fontsize=12)
                    ax.set_ylabel('Stock Price', fontsize=12)
                    ax.legend(fontsize=10)
                    ax.grid(True, alpha=0.3)
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
                    
                    # Plot individual model predictions for better visibility
                    models = {
                        'LSTM': lstm_predictions,
                        'CNN': cnn_predictions,
                        'Hybrid CNN-LSTM': hybrid_predictions
                    }
                    colors = {
                        'LSTM': 'blue',
                        'CNN': 'green',
                        'Hybrid CNN-LSTM': 'red'
                    }
                    
                    st.subheader("Individual Model Performance")
                    for model_name, predictions in models.items():
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Plot predictions
                            fig, ax = plt.subplots(figsize=(10, 6))
                            ax.plot(test_dates, y_test_actual, label='Actual', color='black', linewidth=2)
                            ax.plot(test_dates, predictions, label=f'{model_name} Predictions', 
                                    color=colors[model_name], alpha=0.8)
                            ax.set_title(f'{model_name} Predictions vs Actual', fontsize=14)
                            ax.set_xlabel('Date', fontsize=10)
                            ax.set_ylabel('Stock Price', fontsize=10)
                            ax.legend(fontsize=10)
                            ax.grid(True, alpha=0.3)
                            plt.xticks(rotation=45)
                            st.pyplot(fig)
                            
                        with col2:
                            # Plot error distribution
                            errors = y_test_actual.flatten() - predictions.flatten()
                            fig, ax = plt.subplots(figsize=(10, 6))
                            ax.hist(errors, bins=30, alpha=0.7, color=colors[model_name])
                            ax.axvline(x=0, color='red', linestyle='--', linewidth=1)
                            ax.set_title(f'{model_name} Error Distribution', fontsize=14)
                            ax.set_xlabel('Error (Actual - Predicted)', fontsize=10)
                            ax.set_ylabel('Frequency', fontsize=10)
                            ax.grid(True, alpha=0.3)
                            st.pyplot(fig)
                    
                    # Determine the best model based on RMSE
                    best_model_name = metrics_df.loc[metrics_df['RMSE'].idxmin(), 'Model']
                    # Calculate improvement percentages over other models
                    if best_model_name == 'LSTM':
                        best_rmse = lstm_rmse
                        cnn_improve_pct = ((cnn_rmse - lstm_rmse) / cnn_rmse) * 100
                        hybrid_improve_pct = ((hybrid_rmse - lstm_rmse) / hybrid_rmse) * 100
                        improvement_text = f"LSTM outperforms CNN by {cnn_improve_pct:.2f}% and Hybrid by {hybrid_improve_pct:.2f}% in RMSE."
                    elif best_model_name == 'CNN':
                        best_rmse = cnn_rmse
                        lstm_improve_pct = ((lstm_rmse - cnn_rmse) / lstm_rmse) * 100
                        hybrid_improve_pct = ((hybrid_rmse - cnn_rmse) / hybrid_rmse) * 100
                        improvement_text = f"CNN outperforms LSTM by {lstm_improve_pct:.2f}% and Hybrid by {hybrid_improve_pct:.2f}% in RMSE."
                    else:  # Hybrid CNN-LSTM
                        best_rmse = hybrid_rmse
                        lstm_improve_pct = ((lstm_rmse - hybrid_rmse) / lstm_rmse) * 100
                        cnn_improve_pct = ((cnn_rmse - hybrid_rmse) / cnn_rmse) * 100
                        improvement_text = f"Hybrid outperforms LSTM by {lstm_improve_pct:.2f}% and CNN by {cnn_improve_pct:.2f}% in RMSE."
                    
                    st.success(f"Best performing model based on RMSE: {best_model_name}")
                
                # Future Predictions
                with future_tab:
                    st.write("Generating Future Predictions...")
                    
                    # Get the best model
                    if best_model_name == 'LSTM':
                        best_model = lstm_model
                    elif best_model_name == 'CNN':
                        best_model = cnn_model
                    else:  # Hybrid CNN-LSTM
                        best_model = hybrid_model
                    
                    # Generate future predictions
                    future_predictions = make_future_predictions(best_model, scaled_data, lookback, future_days, scaler)
                    
                    # Generate future dates
                    last_date = data.index[-1]
                    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_days)
                    
                    # Create a DataFrame with future predictions
                    future_df = pd.DataFrame({
                        'Date': future_dates,
                        'Predicted_Price': future_predictions.flatten()
                    })
                    
                    # Display future predictions
                    st.subheader(f"{future_days}-Day Future Predictions")
                    st.dataframe(future_df)
                    
                    # Plot historical and future predictions
                    st.subheader(f"Future Price Prediction for {stock}")
                    fig, ax = plt.subplots(figsize=(12, 6))
                    # Plot historical data (last 365 days or available data)
                    days_to_show = min(365, len(data))
                    ax.plot(data.index[-days_to_show:], data['Close'].values[-days_to_show:], 
                            label='Historical Data', color='black')
                    # Plot future predictions
                    ax.plot(future_dates, future_predictions, 
                            label=f'Future Predictions ({best_model_name})', 
                            color='red', linestyle='--')
                    ax.axvline(x=last_date, color='gray', linestyle='--', label='Prediction Start')
                    ax.set_title(f'{future_days}-Day Future Stock Price Prediction for {stock}', fontsize=14)
                    ax.set_xlabel('Date', fontsize=12)
                    ax.set_ylabel('Stock Price', fontsize=12)
                    ax.legend(fontsize=10)
                    ax.grid(True, alpha=0.3)
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
                    
                    # Get the trend information
                    last_price = data['Close'].iloc[-1]
                    future_last_price = future_predictions[-1][0]
                    price_change = future_last_price - last_price
                    percent_change = (price_change / last_price) * 100
                    
                    # Display trend information
                    st.subheader("Prediction Summary")
                    
                    # Best model performance summary
                    st.write(f"**Best Model Performance:** {best_model_name}")
                    st.write(improvement_text)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Current Price", f"${last_price:.2f}")
                    
                    with col2:
                        st.metric(f"Predicted Price ({future_days} days)", 
                                f"${future_last_price:.2f}", 
                                f"{percent_change:.2f}%")
                    
                    with col3:
                        if percent_change > 0:
                            st.success(f"Upward Trend (+${price_change:.2f})")
                        else:
                            st.error(f"Downward Trend (-${abs(price_change):.2f})")
                    
                    # Additional prediction summary
                    st.subheader("Detailed Prediction Analysis")
                    
                    # Calculate min, max, and average predicted prices
                    min_price = future_predictions.min()
                    max_price = future_predictions.max()
                    avg_price = future_predictions.mean()
                    
                    # Volatility measure (standard deviation)
                    volatility = future_predictions.std()
                    
                    # Price range
                    price_range = max_price - min_price
                    
                    # Create summary metrics
                    summary_col1, summary_col2, summary_col3 = st.columns(3)
                    with summary_col1:
                        st.metric("Min Predicted Price", f"${min_price[0]:.2f}")
                        st.metric("Max Predicted Price", f"${max_price[0]:.2f}")
                    
                    with summary_col2:
                        st.metric("Average Predicted Price", f"${avg_price[0]:.2f}")
                        st.metric("Price Range", f"${price_range[0]:.2f}")
                    
                    with summary_col3:
                        st.metric("Predicted Volatility", f"${volatility[0]:.2f}")
                        # Find the day with the highest predicted price
                        max_day_idx = np.argmax(future_predictions)
                        max_day_date = future_dates[max_day_idx]
                        st.metric("Highest Price Date", max_day_date.strftime('%Y-%m-%d'))
                    
                    # Calculate the data period
                    first_date = data.index[0]
                    data_years = (last_date - first_date).days / 365.25
                    
                    # Display prediction metadata
                    st.info(f"**Analysis Summary:** Prediction for {stock} based on {data_years:.1f} years of historical data (from {first_date.strftime('%Y-%m-%d')} to {last_date.strftime('%Y-%m-%d')}). Forecasting {future_days} days ahead using {best_model_name} model with {lookback}-day lookback period.")
                    
                    # Display disclaimer
                    st.warning("""
                    **Disclaimer**: These predictions are based on historical patterns and should not be the sole basis for investment decisions.
                    Always consider fundamental analysis and market conditions before making investment decisions.
                    """)

            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.error("Please check the ticker symbol and try again.")

# Run the app
if __name__ == "__main__":
    main()