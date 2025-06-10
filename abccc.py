import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import GridSearchCV
import plotly.express as px
import plotly.graph_objects as go

# Updated MaintenancePredictor Class
class MaintenancePredictor:
    def __init__(self):
        """
        Initialize predictive maintenance models
        """
        self.random_forest = RandomForestClassifier(random_state=42)
        self.lstm_model = None
        self.isolation_forest = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)

    def hyperparameter_tuning(self, X_train, y_train):
        """
        Perform Grid Search to optimize Random Forest hyperparameters.
        
        :param X_train: Training feature matrix
        :param y_train: Training target variable
        """
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
        }
        grid_search = GridSearchCV(self.random_forest, param_grid, cv=3, scoring='accuracy', verbose=1)
        grid_search.fit(X_train, y_train)
        self.random_forest = grid_search.best_estimator_

    def train_anomaly_detection(self, X_train):
        """
        Train an Isolation Forest for anomaly detection.
        
        :param X_train: Training feature matrix
        """
        self.isolation_forest.fit(X_train)

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
        from tensorflow.keras.callbacks import EarlyStopping
        X_train_reshaped = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))

        self.lstm_model = Sequential([
            LSTM(50, input_shape=(1, X_train.shape[1]), return_sequences=True),
            Dropout(0.2),
            LSTM(25),
            Dense(1, activation='sigmoid')
        ])

        self.lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        self.lstm_model.fit(
            X_train_reshaped, y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=0
        )

    def visualize_feature_importance(self, feature_names):
        """
        Visualize feature importance from Random Forest
        
        :param feature_names: List of feature names
        """
        importance = self.random_forest.feature_importances_
        plt.figure(figsize=(10, 6))
        sns.barplot(x=importance, y=feature_names)
        plt.title("Feature Importance")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.show()

    def plot_roc_curve(self, X_test, y_test):
        """
        Plot ROC curve for Random Forest and LSTM models
        
        :param X_test: Test feature matrix
        :param y_test: Test target variable
        """
        plt.figure(figsize=(10, 6))

        # Random Forest ROC Curve
        rf_probs = self.random_forest.predict_proba(X_test)[:, 1]
        rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_probs)
        rf_auc = auc(rf_fpr, rf_tpr)
        plt.plot(rf_fpr, rf_tpr, label=f"Random Forest (AUC = {rf_auc:.2f})")

        # LSTM ROC Curve
        if self.lstm_model:
            X_test_reshaped = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
            lstm_probs = self.lstm_model.predict(X_test_reshaped).ravel()
            lstm_fpr, lstm_tpr, _ = roc_curve(y_test, lstm_probs)
            lstm_auc = auc(lstm_fpr, lstm_tpr)
            plt.plot(lstm_fpr, lstm_tpr, label=f"LSTM (AUC = {lstm_auc:.2f})")

        plt.title("ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        plt.show()

    def visualize_sensor_trends(self, df):
        """
        Visualize trends in sensor data using Plotly
        
        :param df: DataFrame containing sensor data
        """
        fig = px.line(
            df, x='timestamp', y=['sensor_reading', 'temperature', 'humidity', 'wind_speed'],
            labels={'value': 'Measurements', 'variable': 'Sensor Types'},
            title="Sensor Data Trends Over Time"
        )
        fig.show()

    def visualize_anomalies(self, df, features):
        """
        Visualize anomalies detected by the Isolation Forest
        
        :param df: Original DataFrame
        :param features: Features used for anomaly detection
        """
        df['anomaly'] = self.isolation_forest.predict(df[features])
        fig = px.scatter(
            df, x='timestamp', y='sensor_reading',
            color='anomaly',
            title="Anomaly Detection in Sensor Data",
            labels={'anomaly': 'Anomaly (1=Normal, -1=Anomalous)'}
        )
        fig.show()
