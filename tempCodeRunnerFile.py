import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import pygame
import speech_recognition as sr
from sklearn.model_selection import train_test_split

# Initialize Pygame
pygame.init()

# Screen dimensions
screen_width = 800
screen_height = 600

# Define the colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)

# Create a Pygame window
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption('Vehicle Simulation')

# Simulate the sensor data
def generate_synthetic_data(num_samples=10000):
    weather_factor = np.random.uniform(0, 1, num_samples)
    speed = np.random.uniform(0, 100, num_samples)
    obstacles = np.random.randint(0, 2, num_samples)
    sensor_health = np.random.uniform(0, 1, num_samples)
    weather_conditions = np.random.randint(0, 2, num_samples)
    
    sensor_failure = (sensor_health < 0.5) & (weather_conditions == 1)
    sensor_failure = sensor_failure.astype(int)
    
    X = np.column_stack([weather_factor, speed, obstacles, sensor_health, weather_conditions])
    y = sensor_failure
    return X, y

# Build the model
def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(5,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Train the model
def train_model(X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model = build_model()
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))
    loss, accuracy = model.evaluate(X_val, y_val)
    print(f"Validation Loss: {loss}, Validation Accuracy: {accuracy}")
    return model

# Speech recognition function for voice commands
def listen_for_commands():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    while True:
        with mic as source:
            print("Listening for voice command...")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)

        try:
            command = recognizer.recognize_google(audio).lower()
            print(f"Command received: {command}")
            return command
        except sr.UnknownValueError:
            print("Sorry, I didn't understand that.")
        except sr.RequestError:
            print("Sorry, there was a problem with the speech service.")

# Real-time simulation logic
def real_time_simulation(model):
    vehicle_position = [screen_width // 2, screen_height // 2]
    vehicle_speed = 5
    vehicle_direction = [1, 0]  # Moving right initially
    is_vehicle_running = False

    while True:
        screen.fill(WHITE)
        
        # Simulate random real-time conditions
        weather_factor = np.random.uniform(0, 1)
        speed = np.random.uniform(0, 100)
        obstacles = np.random.randint(0, 2)
        sensor_health = np.random.uniform(0, 1)
        weather_conditions = np.random.randint(0, 2)
        
        input_data = np.array([[weather_factor, speed, obstacles, sensor_health, weather_conditions]])
        
        # Predict sensor failure or success
        prediction = model.predict(input_data)
        predicted_label = 1 if prediction >= 0.5 else 0
        
        # Update the vehicle status based on prediction
        if predicted_label == 0:
            is_vehicle_running = False
            print("Sensor Failure Detected! Stopping the vehicle.")
        else:
            is_vehicle_running = True
            print("Sensor is functional. Vehicle continues to move.")
        
        if weather_conditions == 1:
            print("Adverse weather detected! Vehicle taking alternate route.")

        # Listen for commands like "Start vehicle" or "Stop vehicle"
        command = listen_for_commands()
        if "start" in command and not is_vehicle_running:
            is_vehicle_running = True
            print("Starting the vehicle...")
        elif "stop" in command and is_vehicle_running:
            is_vehicle_running = False
            print("Stopping the vehicle...")

        # Handle obstacles
        if obstacles == 1:
            print("Obstacle detected! Vehicle stops or changes direction.")
            vehicle_direction = [0, -1]  # Stop the vehicle or take action

        # Update vehicle position
        if is_vehicle_running:
            vehicle_position[0] += vehicle_speed * vehicle_direction[0]
            vehicle_position[1] += vehicle_speed * vehicle_direction[1]

        # Draw the vehicle (as a circle for simplicity)
        pygame.draw.circle(screen, GREEN if is_vehicle_running else RED, vehicle_position, 20)

        # Draw obstacles (as squares)
        if obstacles == 1:
            pygame.draw.rect(screen, YELLOW, (np.random.randint(50, screen_width-50), np.random.randint(50, screen_height-50), 30, 30))

        # Refresh the screen
        pygame.display.flip()

        # Pause for a short time (real-time simulation)
        time.sleep(1)

# Main function to run everything
def main():
    X, y = generate_synthetic_data()
    model = train_model(X, y)
    real_time_simulation(model)

# Run the program
if __name__ == "__main__":
    main() 