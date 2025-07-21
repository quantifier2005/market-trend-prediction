import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.callbacks import EarlyStopping
import plotly.graph_objects as go

# --------------------------
# Helper function for LSTM model
# --------------------------
def build_lstm_model(timesteps, features_per_timestep):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(timesteps, features_per_timestep)))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# --------------------------
# Sidebar Configuration
# --------------------------
st.sidebar.header("Configuration")

# File uploader remains in the main pane for prominence
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

# Checkbox for model comparison
compare_models = st.sidebar.checkbox("Compare Different Models", value=False)

if uploaded_file is not None:
    # Read the dataset
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview:")
    st.write(df.head())

    # Ensure the dataset has a date column named 'observation_date'
    date_column = 'observation_date'
    if date_column not in df.columns:
        st.error(f"'{date_column}' column is not present in the dataset.")
    else:
        # Convert the observation_date column to datetime and extract year
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
        df = df.dropna(subset=[date_column])
        df['observation_date'] = df[date_column].dt.year

        # Sidebar: Number of samples and years
        num_rows = st.sidebar.slider("Select number of samples to be used", 100, len(df), min(1000, len(df)))
        num_years = st.sidebar.slider(
            "Select number of years for the model",
            1,
            int(df['observation_date'].max() - df['observation_date'].min()),
            5
        )

        # Sidebar: Model selection and hyperparameters
        model_name = st.sidebar.selectbox("Select Model", ["Random Forest", "Support Vector Machine", "XGBoost", "LSTM"])

        if model_name == "LSTM":
            epochs = st.sidebar.number_input("Number of epochs", min_value=1, max_value=1000, value=100)
            batch_size = st.sidebar.number_input("Batch size", min_value=1, max_value=256, value=32)

        # --------------------------
        # Data Preprocessing
        # --------------------------
        # Randomly sample rows
        df = df.sample(n=num_rows, random_state=42)

        # Set target variable
        target = 'inflation_rate'

        # Convert non-numeric columns to numeric or drop them
        df = df.apply(pd.to_numeric, errors='coerce')
        df = df.dropna(axis=1, how='any')  # Drop columns with NaN values

        # Select specific columns for correlation
        selected_columns = ['inflation_rate', 'GDP', 'CPI', 'unemployment', 'DGS30_x',
                            'Yield Spread', 'IND US', 'US UK', 'US RUSSIA',
                            'Interest Rate', 'DAAA', 'DBAA', 'DFF', 'Sentiment',
                            'DGS30_y', 'DGS10', 'DGS3MO', 'DGS2']
        selected_columns = [col for col in selected_columns if col in df.columns]

        # Create correlation matrix and heatmap
        corr_matrix = df[selected_columns].corr()
        st.write("### Correlation Heatmap:")
        fig_heat, ax_heat = plt.subplots(figsize=(12, 12))
        sns.heatmap(corr_matrix, annot=True, cmap='BuPu', ax=ax_heat, annot_kws={"size": 10}, fmt=".2f", linewidths=.5)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(fontsize=10)
        plt.title('Correlation Heatmap', fontsize=15)
        plt.tight_layout()
        st.pyplot(fig_heat)

        # Drop least correlated feature for the target variable
        least_correlated_feature = corr_matrix[target].drop(target).idxmin()
        st.write(f"**Dropping least correlated feature:** {least_correlated_feature}")
        df = df.drop(columns=[least_correlated_feature])

        # Use all features except the target variable
        features = df.columns.drop(target).tolist()

        # Filter dataset based on the selected number of years
        max_year = df['observation_date'].max()
        min_year = max_year - num_years + 1
        df_filtered = df[(df['observation_date'] >= min_year) & (df['observation_date'] <= max_year)]
        st.write("### Filtered Dataset Preview:")
        st.write(df_filtered.head())
        st.write(f"Filtered dataset size: {df_filtered.shape}")

        if df_filtered.empty:
            st.error("Filtered dataset is empty. Try selecting more years or samples.")
        else:
            # --------------------------
            # Train-test Split and Scaling
            # --------------------------
            X = df_filtered[features]
            y = df_filtered[target]
            st.write(f"**X shape:** {X.shape}, **y shape:** {y.shape}")

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.10, random_state=42)

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)
            X_test = scaler.transform(X_test)

            # Save copies of 2D data for the classical models in the comparison section
            X_train_orig = X_train.copy()
            X_test_orig = X_test.copy()
            X_val_orig = X_val.copy()

            # --------------------------
            # Model Initialization and Training
            # --------------------------
            st.write("### Training the model...")
            if model_name == "LSTM":
                # Reshape data for LSTM:
                num_features = X_train.shape[1]
                timesteps = max(10, num_features)  # Use at least 10 timesteps
                features_per_timestep = num_features // timesteps

                if num_features % timesteps == 0:
                    X_train = X_train.reshape(X_train.shape[0], timesteps, features_per_timestep)
                    X_val = X_val.reshape(X_val.shape[0], timesteps, features_per_timestep)
                    X_test = X_test.reshape(X_test.shape[0], timesteps, features_per_timestep)
                else:
                    st.error(f"Number of features ({num_features}) is not divisible by timesteps ({timesteps}). Try reducing features.")
                    st.stop()

                # Build LSTM model for main training
                model = build_lstm_model(timesteps, features_per_timestep)
                early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

                with st.spinner("Training LSTM..."):
                    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                                         epochs=epochs, batch_size=batch_size, shuffle=True, callbacks=[early_stopping])
                st.write("### LSTM Training History")
                history_df = pd.DataFrame(history.history)
                st.line_chart(history_df)

                y_val_pred = model.predict(X_val).flatten()
                y_test_pred = model.predict(X_test).flatten()
            else:
                if model_name == "Random Forest":
                    model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
                elif model_name == "Support Vector Machine":
                    model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
                elif model_name == "XGBoost":
                    model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1,
                                         objective='reg:squarederror', random_state=42)

                with st.spinner(f"Training {model_name}..."):
                    model.fit(X_train, y_train)
                y_val_pred = model.predict(X_val)
                y_test_pred = model.predict(X_test)

            # --------------------------
            # Calculate and Display Metrics
            # --------------------------
            mae_val = mean_absolute_error(y_val, y_val_pred)
            mse_val = mean_squared_error(y_val, y_val_pred)
            r2_val = r2_score(y_val, y_val_pred)

            mae_test = mean_absolute_error(y_test, y_test_pred)
            mse_test = mean_squared_error(y_test, y_test_pred)
            r2_test = r2_score(y_test, y_test_pred)

            metrics_data = {
                'Metric': ['MAE', 'MSE', 'R²'],
                'Validation': [mae_val, mse_val, r2_val],
                'Test': [mae_test, mse_test, r2_test]
            }
            metrics_df = pd.DataFrame(metrics_data)
            st.write("### Model Metrics")
            st.table(metrics_df)

            # --------------------------
            # Feature Importance (for tree models)
            # --------------------------
            if model_name in ["Random Forest", "XGBoost"]:
                importance = model.feature_importances_
                imp_df = pd.DataFrame({'Feature': features, 'Importance': importance})
                imp_df = imp_df.sort_values('Importance', ascending=False)
                st.write("### Feature Importances")
                st.bar_chart(imp_df.set_index('Feature'))

            # --------------------------
            # Visualization: Predicted vs Actual
            # --------------------------
            fig_scatter = go.Figure()
            fig_scatter.add_trace(go.Scatter(x=list(range(len(y_test))), y=y_test,
                                             mode='markers',
                                             marker=dict(color='blue', size=8, opacity=0.7),
                                             name='Actual Values'))
            fig_scatter.add_trace(go.Scatter(x=list(range(len(y_test_pred))), y=y_test_pred,
                                             mode='markers',
                                             marker=dict(color='orange', size=8, opacity=0.7),
                                             name='Predicted Values'))
            fig_scatter.update_layout(
                title=f"{model_name} Predicted vs Actual Values",
                xaxis_title="Sample Index",
                yaxis_title="Inflation Rate",
                legend_title="Legend",
                template="plotly_white"
            )
            st.write("### Predicted vs Actual Scatter Plot")
            st.plotly_chart(fig_scatter)

            # --------------------------
            # Download Predictions
            # --------------------------
            results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_test_pred})
            csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Predictions", csv, "predictions.csv", "text/csv")

            # --------------------------
            # Model Comparison Section (only when requested)
            # --------------------------
            if compare_models:
                st.write("## Model Comparison on Test Set (Classical ML Models + LSTM)")
                comp_metrics = []
                # --- Classical models using original 2D data ---
                comparison_models = {
                    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10),
                    "Support Vector Machine": SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1),
                    "XGBoost": XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1,
                                            objective='reg:squarederror', random_state=42)
                }
                for name, comp_model in comparison_models.items():
                    with st.spinner(f"Training {name} for comparison..."):
                        comp_model.fit(X_train_orig, y_train)
                    y_test_pred_comp = comp_model.predict(X_test_orig)
                    mae_comp = mean_absolute_error(y_test, y_test_pred_comp)
                    mse_comp = mean_squared_error(y_test, y_test_pred_comp)
                    r2_comp = r2_score(y_test, y_test_pred_comp)
                    comp_metrics.append({
                        "Model": name,
                        "MAE": mae_comp,
                        "MSE": mse_comp,
                        "R²": r2_comp
                    })

                # --- LSTM Model for Comparison ---
                num_features_cmp = X_train_orig.shape[1]
                lstm_timesteps = max(10, num_features_cmp)
                if num_features_cmp % lstm_timesteps == 0:
                    lstm_features_per_timestep = num_features_cmp // lstm_timesteps
                    X_train_lstm = X_train_orig.reshape(X_train_orig.shape[0], lstm_timesteps, lstm_features_per_timestep)
                    X_test_lstm = X_test_orig.reshape(X_test_orig.shape[0], lstm_timesteps, lstm_features_per_timestep)
                    
                    # Use the same LSTM architecture. For comparison, use default epochs and batch size.
                    epochs_lstm = epochs if model_name=="LSTM" else 50
                    batch_size_lstm = batch_size if model_name=="LSTM" else 32
                    
                    with st.spinner("Training LSTM for comparison..."):
                        lstm_model = build_lstm_model(lstm_timesteps, lstm_features_per_timestep)
                        lstm_model.fit(X_train_lstm, y_train, epochs=epochs_lstm,
                                       batch_size=batch_size_lstm, verbose=0)
                    y_test_pred_lstm = lstm_model.predict(X_test_lstm).flatten()
                    mae_lstm = mean_absolute_error(y_test, y_test_pred_lstm)
                    mse_lstm = mean_squared_error(y_test, y_test_pred_lstm)
                    r2_lstm = r2_score(y_test, y_test_pred_lstm)
                    comp_metrics.append({
                        "Model": "LSTM",
                        "MAE": mae_lstm,
                        "MSE": mse_lstm,
                        "R²": r2_lstm
                    })
                else:
                    st.warning("Skipping LSTM in comparison due to feature shape mismatch (not divisible by desired timesteps).")
                
                comp_df = pd.DataFrame(comp_metrics)
                st.table(comp_df)



