import random
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import tensorflow as tf
from flask import Flask, jsonify, render_template
import threading
import time
import pygame

# Initialize Pygame
pygame.init()

# Constants for the simulation
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
VEHICLE_WIDTH = 50
VEHICLE_HEIGHT = 30
TRACK_COLOR = (50, 50, 50)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
ALERT_COLOR = (255, 69, 0)  # Red for alert
GREY = (169, 169, 169)
SNOW_COLOR = (240, 240, 240)
RAINDROP_COLOR = (0, 191, 255)
BLUE = (0, 0, 255)
TRACK_MARK_COLOR = (255, 255, 255)

# Alert Sound
pygame.mixer.init()
ALERT_SOUND = "alarm-26718.mp3"
pygame.mixer.music.load(ALERT_SOUND)

# Flask App Initialization
app = Flask(__name__)
current_data = {
    "environment": "Clear",
    "sensor_condition": "Normal",
    "self_cleaning_active": False,
    "sensor_data": {},
    "current_speed": 0
}

# Generate synthetic data
def generate_sample_data(num_rows=1000):
    data = {
        'sensor_reading': np.random.randint(40, 100, size=num_rows),
        'ambient_temp': np.random.randint(20, 40, size=num_rows),
        'humidity_level': np.random.randint(50, 80, size=num_rows),
        'wind_velocity': np.random.randint(5, 20, size=num_rows),
        'failure_indicator': np.random.choice([0, 1], size=num_rows)
    }
    return pd.DataFrame(data)

# Train a predictive maintenance model using TensorFlow (Deep Learning)
def train_prediction_model():
    data = generate_sample_data()
    features = ['sensor_reading', 'ambient_temp', 'humidity_level', 'wind_velocity']
    X = data[features]
    y = data['failure_indicator']

    # Normalize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, 'scaler_model.pkl')

    # Build the TensorFlow model
    model = tf.keras.Sequential([ 
        tf.keras.layers.Dense(64, activation='relu', input_dim=X_scaled.shape[1]),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer with sigmoid for binary classification
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_scaled, y, epochs=20, batch_size=32, validation_split=0.2)

    # Save the trained model
    model.save('maintenance_prediction_model.h5')

# Load the predictive maintenance model
def load_prediction_model():
    model = tf.keras.models.load_model('maintenance_prediction_model.h5')
    scaler = joblib.load('scaler_model.pkl')
    return model, scaler

# Environment simulation
def simulate_environment_conditions():
    conditions = ['Clear', 'Rain', 'Fog', 'Snow']
    return random.choice(conditions)

# Self-cleaning mode
def activate_self_cleaning(weather):
    return weather in ['Rain', 'Snow']

# Flask route: Current data
@app.route('/data', methods=['GET'])
def get_current_data():
    return jsonify(current_data)

@app.route('/')
def home_dashboard():
    print("Home route accessed")  # Add print statement for debugging
    try:
        return render_template('home.html')
    except Exception as e:
        print(f"Error rendering template: {e}")
        return f"Error: {e}", 500  # Return error details



# Main vehicle simulation
def vehicle_simulation():
    train_prediction_model()  # Train the prediction model
    model, scaler = load_prediction_model()  # Load the trained model

    # Initialize simulation
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Real World Vehicle Simulation")
    clock = pygame.time.Clock()

    vehicle_x = SCREEN_WIDTH // 2 - VEHICLE_WIDTH // 2
    vehicle_y = SCREEN_HEIGHT - 100
    vehicle_speed = 10
    track_y = SCREEN_HEIGHT - 50  # Track's y position
    running = True
    sensor_failure_start_time = None
    cleaning_in_progress = False

    while running:
        screen.fill(TRACK_COLOR)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Simulate environment conditions
        environment = simulate_environment_conditions()
        environment_color = GREY if environment == 'Fog' else SNOW_COLOR if environment == 'Snow' else RAINDROP_COLOR if environment == 'Rain' else TRACK_COLOR
        pygame.draw.rect(screen, environment_color, (0, 0, SCREEN_WIDTH, SCREEN_HEIGHT // 4))

        # Generate sensor data
        sensor_data = {
            'sensor_reading': random.randint(40, 100),
            'ambient_temp': random.randint(20, 40),
            'humidity_level': random.randint(50, 80),
            'wind_velocity': random.randint(5, 20)
        }

        # Predict sensor health using the model
        sensor_df = pd.DataFrame([sensor_data])
        scaled_data = scaler.transform(sensor_df)
        prediction = model.predict(scaled_data)

        # Update Flask data
        current_data.update({
            "environment": environment,
            "sensor_data": sensor_data
        })

        # Check sensor condition and handle failure
        if prediction >= 0.5:  # Failure predicted
            if sensor_failure_start_time is None:
                sensor_failure_start_time = time.time()
            else:
                # If the sensor failure has been ongoing for more than 5 seconds, activate the cleaning process
                if time.time() - sensor_failure_start_time > 5 and not cleaning_in_progress:
                    current_data["sensor_condition"] = "Cleaning Activated"
                    cleaning_in_progress = True
                    # Simulate self-cleaning for a brief period
                    time.sleep(2)
                    cleaning_in_progress = False
                    sensor_failure_start_time = None  # Reset failure time after cleaning
                else:
                    current_data["sensor_condition"] = "Failure Predicted"
                    vehicle_speed = 0
                    pygame.mixer.music.play()
                    pygame.draw.rect(screen, ALERT_COLOR, (vehicle_x, vehicle_y, VEHICLE_WIDTH, VEHICLE_HEIGHT))
        else:
            # Normal sensor operation
            current_data["sensor_condition"] = "Normal"
            vehicle_speed = 10
            pygame.draw.rect(screen, BLUE, (vehicle_x, vehicle_y, VEHICLE_WIDTH, VEHICLE_HEIGHT))

        # Display track markings
        pygame.draw.line(screen, TRACK_MARK_COLOR, (SCREEN_WIDTH // 2, track_y), (SCREEN_WIDTH // 2, 0), 5)
        pygame.draw.line(screen, TRACK_MARK_COLOR, (SCREEN_WIDTH // 2 - 200, track_y), (SCREEN_WIDTH // 2 - 200, 0), 5)
        pygame.draw.line(screen, TRACK_MARK_COLOR, (SCREEN_WIDTH // 2 + 200, track_y), (SCREEN_WIDTH // 2 + 200, 0), 5)

        # Display dashboard with vehicle speed, sensor status, and alert
        font = pygame.font.SysFont(None, 30)
        text = font.render(f"Speed: {current_data['current_speed']} km/h", True, WHITE)
        screen.blit(text, (10, 10))

        status_text = font.render(f"Sensor Status: {current_data['sensor_condition']}", True, WHITE)
        screen.blit(status_text, (10, 40))

        # Move the vehicle
        vehicle_y -= vehicle_speed
        if vehicle_y < 0:
            vehicle_y = SCREEN_HEIGHT - 100

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()

if __name__ == "__main__":
    # Start vehicle simulation in a separate thread
    simulation_thread = threading.Thread(target=vehicle_simulation)
    simulation_thread.start()

    # Run the Flask app in the main thread
    app.run(debug=True, use_reloader=False) 