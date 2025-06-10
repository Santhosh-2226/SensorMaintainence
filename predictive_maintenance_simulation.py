import pygame
import random
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from flask import Flask, jsonify, render_template
import threading

# Initialize Pygame
pygame.init()

# Constants for the simulation
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
CAR_WIDTH = 50
CAR_HEIGHT = 30
ROAD_COLOR = (50, 50, 50)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
WARNING_COLOR = (255, 69, 0)
GREY = (169, 169, 169)
SNOW_COLOR = (240, 240, 240)
RAINDROP_COLOR = (0, 191, 255)
BLUE = (0, 0, 255)
ROAD_MARK_COLOR = (255, 255, 255)

# Alarm Sound
pygame.mixer.init()
ALARM_SOUND = "C:\\Users\\santh\\Downloads\\alarm-26718.mp3"
pygame.mixer.music.load(ALARM_SOUND)

# Flask App Initialization
app = Flask(__name__)
real_time_data = {
    "weather": "Clear",
    "sensor_status": "Healthy",
    "self_cleaning": False,
    "sensor_readings": {},
    "speed": 0
}

# Generate synthetic data
def generate_data(num_rows=1000):
    data = {
        'reading': np.random.randint(40, 100, size=num_rows),
        'temperature': np.random.randint(20, 40, size=num_rows),
        'humidity': np.random.randint(50, 80, size=num_rows),
        'wind_speed': np.random.randint(5, 20, size=num_rows),
        'failure_label': np.random.choice([0, 1], size=num_rows)
    }
    return pd.DataFrame(data)

# Train ML model
def train_model():
    df = generate_data()
    features = ['reading', 'temperature', 'humidity', 'wind_speed']
    X = df[features]
    y = df['failure_label']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, 'scaler.pkl')

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    joblib.dump(model, 'sensor_health_model.pkl')
    print(classification_report(y_test, model.predict(X_test)))

# Load ML model
def load_model():
    model = joblib.load('sensor_health_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

# Weather simulation
def simulate_weather():
    weather_conditions = ['Clear', 'Rain', 'Fog', 'Snow']
    return random.choice(weather_conditions)

# Self-cleaning mode
def self_cleaning_mode(weather):
    return weather in ['Rain', 'Snow']

# Flask route: Real-time data
@app.route('/data', methods=['GET'])
def get_real_time_data():
    return jsonify(real_time_data)

# Flask route: Dashboard
@app.route('/')
def dashboard():
    return render_template('dashboard.html')

# Main car simulation
def car_simulation():
    train_model()
    model, scaler = load_model()

    # Initialize simulation
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Real World Autonomous Vehicle Simulation")
    clock = pygame.time.Clock()

    car_x = SCREEN_WIDTH // 2 - CAR_WIDTH // 2
    car_y = SCREEN_HEIGHT - 100
    car_speed = 10
    road_y = SCREEN_HEIGHT - 50  # Road's y position
    running = True

    while running:
        screen.fill(ROAD_COLOR)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Simulate weather
        weather = simulate_weather()
        weather_color = GREY if weather == 'Fog' else SNOW_COLOR if weather == 'Snow' else RAINDROP_COLOR if weather == 'Rain' else ROAD_COLOR
        pygame.draw.rect(screen, weather_color, (0, 0, SCREEN_WIDTH, SCREEN_HEIGHT // 4))

        # Generate sensor data
        sensor_data = {
            'reading': random.randint(40, 100),
            'temperature': random.randint(20, 40),
            'humidity': random.randint(50, 80),
            'wind_speed': random.randint(5, 20)
        }

        # Predict sensor health
        sensor_df = pd.DataFrame([sensor_data])
        scaled_data = scaler.transform(sensor_df)
        prediction = model.predict(scaled_data)[0]

        # Update Flask data
        real_time_data.update({
            "weather": weather,
            "sensor_status": "Healthy" if prediction == 0 else "Failure Predicted",
            "self_cleaning": self_cleaning_mode(weather),
            "sensor_readings": sensor_data,
            "speed": car_speed if prediction == 0 else 0
        })

        # Draw the car
        if prediction == 1:  # Failure detected
            pygame.mixer.music.play()
            car_speed = 0
            pygame.draw.rect(screen, WARNING_COLOR, (car_x, car_y, CAR_WIDTH, CAR_HEIGHT))
        else:
            pygame.draw.rect(screen, BLUE, (car_x, car_y, CAR_WIDTH, CAR_HEIGHT))

        # Display road markings
        pygame.draw.line(screen, ROAD_MARK_COLOR, (SCREEN_WIDTH // 2, road_y), (SCREEN_WIDTH // 2, 0), 5)
        pygame.draw.line(screen, ROAD_MARK_COLOR, (SCREEN_WIDTH // 2 - 200, road_y), (SCREEN_WIDTH // 2 - 200, 0), 5)
        pygame.draw.line(screen, ROAD_MARK_COLOR, (SCREEN_WIDTH // 2 + 200, road_y), (SCREEN_WIDTH // 2 + 200, 0), 5)

        # Move the car
        car_y -= car_speed
        if car_y < 0:
            car_y = SCREEN_HEIGHT - 100

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()

if __name__ == "__main__":
    threading.Thread(target=car_simulation).start()
    app.run(debug=True)
