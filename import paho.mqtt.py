import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta

# Machine Learning and Data Processing Libraries
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Deep Learning Libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def generate_synthetic_sensor_data(num_samples=1000):
    """
    Generate synthetic sensor data for predictive maintenance
    
    :param num_samples: Number of data points to generate
    :return: Pandas DataFrame with synthetic sensor measurements
    """
    np.random.seed(42)
    
    # Generate timestamps
    start_date = datetime(2023, 1, 1)
    timestamps = [start_date + timedelta(hours=i) for i in range(num_samples)]
    
    # Generate features with some realistic variations
    temperature = np.random.normal(25, 10, num_samples)  # Mean 25°C, std dev 10
    humidity = np.clip(np.random.normal(50, 15, num_samples), 10, 90)  # 10-90% range
    wind_speed = np.abs(np.random.normal(5, 3, num_samples))  # Always positive
    
    # Sensor reading with some correlation to other features
    sensor_reading = (
        temperature * 0.5 + 
        humidity * 0.3 + 
        wind_speed * 0.2 + 
        np.random.normal(0, 10, num_samples)
    )
    
    # Maintenance needed based on sensor reading and other conditions
    maintenance_threshold = np.percentile(sensor_reading, 75)
    maintenance_needed = (
        (sensor_reading > maintenance_threshold) | 
        (temperature > 35) | 
        (temperature < 0) | 
        (humidity > 80) | 
        (wind_speed > 15)
    ).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'sensor_reading': sensor_reading,
        'temperature': temperature,
        'humidity': humidity,
        'wind_speed': wind_speed,
        'maintenance_needed': maintenance_needed
    })
    
    return df

class SensorDataPreprocessor:
    def __init__(self, weather_api_key=None):
        """
        Initialize the data preprocessor with optional weather API integration
        
        :param weather_api_key: API key for weather data retrieval
        """
        self.weather_api_key = weather_api_key
        self.scaler = StandardScaler()
    
    def load_sensor_data(self, filepath):
        """
        Load sensor data from CSV or JSON
        
        :param filepath: Path to the sensor data file
        :return: Pandas DataFrame with sensor measurements
        """
        try:
            if filepath.endswith('.csv'):
                return pd.read_csv(filepath)
            elif filepath.endswith('.json'):
                return pd.read_json(filepath)
            else:
                raise ValueError("Unsupported file format. Use CSV or JSON.")
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def clean_data(self, df):
        """
        Clean and preprocess sensor data
        
        :param df: Input DataFrame
        :return: Cleaned and preprocessed DataFrame
        """
        # Remove rows with missing critical values
        df_cleaned = df.dropna(subset=['sensor_reading', 'temperature', 'humidity'])
        
        # Handle outliers using IQR method
        Q1 = df_cleaned['sensor_reading'].quantile(0.25)
        Q3 = df_cleaned['sensor_reading'].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        df_cleaned = df_cleaned[
            (df_cleaned['sensor_reading'] >= lower_bound) & 
            (df_cleaned['sensor_reading'] <= upper_bound)
        ]
        
        return df_cleaned
    
    def extract_features(self, df):
        """
        Extract and engineer features for machine learning
        
        :param df: Cleaned DataFrame
        :return: Feature matrix and target variable
        """
        # Create time-based features
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
        
        # Feature engineering for weather conditions
        df['weather_risk_score'] = (
            df['temperature'].abs() * 0.5 + 
            df['humidity'] * 0.3 + 
            df['wind_speed'] * 0.2
        )
        
        # Select features for prediction
        features = [
            'sensor_reading', 'temperature', 'humidity', 
            'wind_speed', 'hour', 'day_of_week', 'weather_risk_score'
        ]
        
        X = df[features]
        y = df['maintenance_needed']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y

class MaintenancePredictor:
    def __init__(self):
        """
        Initialize predictive maintenance models
        """
        self.random_forest = RandomForestClassifier(
            n_estimators=100, 
            random_state=42
        )
        self.lstm_model = None
    
    def train_random_forest(self, X_train, y_train):
        """
        Train Random Forest model for maintenance prediction
        
        :param X_train: Training feature matrix
        :param y_train: Training target variable
        """
        self.random_forest.fit(X_train, y_train)
    
    def train_lstm(self, X_train, y_train):
        """
        Train LSTM model for time-series maintenance prediction
        
        :param X_train: Time-series training data
        :param y_train: Training target variable
        """
        # Reshape input for LSTM (samples, time steps, features)
        X_train_reshaped = X_train.reshape(
            (X_train.shape[0], 1, X_train.shape[1])
        )
        
        self.lstm_model = Sequential([
            LSTM(50, input_shape=(1, X_train.shape[1]), return_sequences=True),
            Dropout(0.2),
            LSTM(25),
            Dense(1, activation='sigmoid')
        ])
        
        self.lstm_model.compile(
            optimizer='adam', 
            loss='binary_crossentropy', 
            metrics=['accuracy']
        )
        
        self.lstm_model.fit(
            X_train_reshaped, y_train, 
            epochs=50, 
            batch_size=32, 
            validation_split=0.2,
            verbose=0  # Reduce output verbosity
        )
    
    def evaluate_models(self, X_test, y_test):
        """
        Evaluate model performance
        
        :param X_test: Test feature matrix
        :param y_test: Test target variable
        :return: Dictionary of performance metrics
        """
        results = {}
        
        # Random Forest evaluation
        rf_predictions = self.random_forest.predict(X_test)
        results['random_forest'] = {
            'classification_report': classification_report(y_test, rf_predictions),
            'confusion_matrix': confusion_matrix(y_test, rf_predictions)
        }
        
        # LSTM evaluation
        if self.lstm_model:
            X_test_reshaped = X_test.reshape(
                (X_test.shape[0], 1, X_test.shape[1])
            )
            lstm_predictions = (
                self.lstm_model.predict(X_test_reshaped) > 0.5
            ).astype(int)
            
            results['lstm'] = {
                'classification_report': classification_report(y_test, lstm_predictions),
                'confusion_matrix': confusion_matrix(y_test, lstm_predictions)
            }
        
        return results
    
    def predict_maintenance_need(self, sensor_data):
        """
        Predict maintenance needs for new sensor data
        
        :param sensor_data: New sensor measurements
        :return: Maintenance prediction probabilities
        """
        rf_prob = self.random_forest.predict_proba(sensor_data)
        lstm_prob = None
        
        if self.lstm_model:
            sensor_data_reshaped = sensor_data.reshape(
                (sensor_data.shape[0], 1, sensor_data.shape[1])
            )
            lstm_prob = self.lstm_model.predict(sensor_data_reshaped)
        
        return {
            'random_forest_prob': rf_prob,
            'lstm_prob': lstm_prob
        }

class WeatherRiskAssessment:
    def __init__(self, api_key):
        """
        Initialize weather risk assessment with OpenWeather API
        
        :param api_key: OpenWeather API key
        """
        self.api_key = api_key
        self.base_url = "http://api.openweathermap.org/data/2.5/forecast"
    
    def get_weather_forecast(self, latitude, longitude):
        """
        Retrieve weather forecast for specified location
        
        :param latitude: Geographical latitude
        :param longitude: Geographical longitude
        :return: Parsed weather forecast data
        """
        params = {
            'lat': latitude,
            'lon': longitude,
            'appid': self.api_key,
            'units': 'metric'  # Use Celsius for temperature
        }
        
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Weather API Error: {e}")
            return None
    
    def calculate_weather_risk(self, forecast_data):
        """
        Calculate weather risk based on forecast data
        
        :param forecast_data: Weather forecast from OpenWeather API
        :return: DataFrame with weather risk assessment
        """
        if not forecast_data or 'list' not in forecast_data:
            return None
        
        risk_data = []
        
        for entry in forecast_data['list']:
            timestamp = datetime.fromtimestamp(entry['dt'])
            
            # Risk factors
            temperature = entry['main']['temp']
            humidity = entry['main']['humidity']
            wind_speed = entry['wind']['speed']
            weather_condition = entry['weather'][0]['main']
            
            # Risk scoring logic
            temperature_risk = self._assess_temperature_risk(temperature)
            humidity_risk = self._assess_humidity_risk(humidity)
            wind_risk = self._assess_wind_risk(wind_speed)
            condition_risk = self._assess_condition_risk(weather_condition)
            
            # Composite risk score
            total_risk_score = (
                temperature_risk * 0.3 + 
                humidity_risk * 0.2 + 
                wind_risk * 0.3 + 
                condition_risk * 0.2
            )
            
            risk_data.append({
                'timestamp': timestamp,
                'temperature': temperature,
                'humidity': humidity,
                'wind_speed': wind_speed,
                'weather_condition': weather_condition,
                'risk_score': total_risk_score,
                'maintenance_recommendation': self._get_maintenance_recommendation(total_risk_score)
            })
        
        return pd.DataFrame(risk_data)
    
    def _assess_temperature_risk(self, temperature):
        """
        Assess risk based on temperature extremes
        
        :param temperature: Temperature in Celsius
        :return: Temperature risk score (0-1)
        """
        if temperature < -10 or temperature > 40:
            return 1.0
        elif temperature < 0 or temperature > 35:
            return 0.7
        elif temperature < 5 or temperature > 30:
            return 0.4
        return 0.1
    
    def _assess_humidity_risk(self, humidity):
        """
        Assess risk based on humidity levels
        
        :param humidity: Humidity percentage
        :return: Humidity risk score (0-1)
        """
        if humidity > 90 or humidity < 20:
            return 1.0
        elif humidity > 80 or humidity < 30:
            return 0.7
        return 0.2
    
    def _assess_wind_risk(self, wind_speed):
        """
        Assess risk based on wind speed
        
        :param wind_speed: Wind speed in m/s
        :return: Wind risk score (0-1)
        """
        if wind_speed > 20:
            return 1.0
        elif wind_speed > 10:
            return 0.7
        return 0.2
    
    def _assess_condition_risk(self, condition):
        """
        Assess risk based on weather condition
        
        :param condition: Weather condition string
        :return: Condition risk score (0-1)
        """
        high_risk_conditions = ['Thunderstorm', 'Hurricane', 'Tornado']
        moderate_risk_conditions = ['Rain', 'Snow', 'Extreme']
        
        if condition in high_risk_conditions:
            return 1.0
        elif condition in moderate_risk_conditions:
            return 0.7
        return 0.2
    
    def _get_maintenance_recommendation(self, risk_score):
        """
        Generate maintenance recommendation based on risk score
        
        :param risk_score: Calculated risk score
        :return: Maintenance recommendation string
        """
        if risk_score > 0.8:
            return "Immediate Maintenance Required"
        elif risk_score > 0.5:
            return "Preventive Maintenance Recommended"
        else:
            return "Normal Monitoring Sufficient"

def main():
    """
    Main function to demonstrate the integrated predictive maintenance system
    """
    # OpenWeather API key (replace with your actual API key or use a placeholder)
    WEATHER_API_KEY = "59e6baf3e102058e69d2cf52046c86e9"
    
    # Initialize components
    preprocessor = SensorDataPreprocessor(WEATHER_API_KEY)
    predictor = MaintenancePredictor()
    
    # Generate synthetic sensor data instead of loading from file
    sensor_data = generate_synthetic_sensor_data(num_samples=2000)
    
    # Rest of the main function
    if sensor_data is not None:
        # Clean the data
        cleaned_data = preprocessor.clean_data(sensor_data)
        
        # Extract features
        X, y = preprocessor.extract_features(cleaned_data)
        
        # Split data for training and testing
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train models
        predictor.train_random_forest(X_train, y_train)
        predictor.train_lstm(X_train, y_train)
        
        # Evaluate models
        model_results = predictor.evaluate_models(X_test, y_test)
        print("Model Performance Results:")
        for model_name, results in model_results.items():
            print(f"\n{model_name.replace('_', ' ').title()} Model:")
            print("Classification Report:")
            print(results['classification_report'])
            print("\nConfusion Matrix:")
            print(results['confusion_matrix'])
        
        # Weather Risk Assessment
        # Weather Risk Assessment
        weather_risk = WeatherRiskAssessment(WEATHER_API_KEY)
        
        # Example coordinates (you can replace with actual location coordinates)
        latitude = 40.7128  # New York City latitude
        longitude = -74.0060  # New York City longitude
        
        try:
            # Fetch weather forecast
            forecast_data = weather_risk.get_weather_forecast(latitude, longitude)
            
            if forecast_data:
                # Calculate weather risk
                risk_assessment = weather_risk.calculate_weather_risk(forecast_data)
                
                if risk_assessment is not None:
                    print("\nWeather Risk Assessment:")
                    print(risk_assessment)
                    
                    # Identify high-risk periods
                    high_risk_periods = risk_assessment[
                        risk_assessment['risk_score'] > 0.5
                    ]
                    
                    if not high_risk_periods.empty:
                        print("\nHigh-Risk Maintenance Periods:")
                        print(high_risk_periods[['timestamp', 'risk_score', 'maintenance_recommendation']])
                    else:
                        print("\nNo high-risk maintenance periods detected.")
            else:
                print("Could not retrieve weather forecast data.")
        
        except Exception as e:
            print(f"An error occurred during weather risk assessment: {e}")

if __name__ == "__main__":
    main()