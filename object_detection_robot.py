
import cv2
import numpy as np
import tensorflow as tf
import RPi.GPIO as GPIO
import time

# Load the TensorFlow model
model_url = 'https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2'
model = tf.saved_model.load(model_url)

# GPIO Pin Setup for Raspberry Pi
GPIO.setmode(GPIO.BCM)

# Car motor control pins
MOTOR_LEFT_FORWARD = 17
MOTOR_LEFT_BACKWARD = 27
MOTOR_RIGHT_FORWARD = 22
MOTOR_RIGHT_BACKWARD = 23

# Robotic arm control pin
ARM_PICK_PIN = 18

# Setup GPIO pins
GPIO.setup(MOTOR_LEFT_FORWARD, GPIO.OUT)
GPIO.setup(MOTOR_LEFT_BACKWARD, GPIO.OUT)
GPIO.setup(MOTOR_RIGHT_FORWARD, GPIO.OUT)
GPIO.setup(MOTOR_RIGHT_BACKWARD, GPIO.OUT)
GPIO.setup(ARM_PICK_PIN, GPIO.OUT)

# Define functions for motor and arm control
def move_forward():
    GPIO.output(MOTOR_LEFT_FORWARD, True)
    GPIO.output(MOTOR_RIGHT_FORWARD, True)
    GPIO.output(MOTOR_LEFT_BACKWARD, False)
    GPIO.output(MOTOR_RIGHT_BACKWARD, False)

def stop_car():
    GPIO.output(MOTOR_LEFT_FORWARD, False)
    GPIO.output(MOTOR_RIGHT_FORWARD, False)
    GPIO.output(MOTOR_LEFT_BACKWARD, False)
    GPIO.output(MOTOR_RIGHT_BACKWARD, False)

def pick_object():
    arm_servo = GPIO.PWM(ARM_PICK_PIN, 50)  # Set PWM frequency to 50Hz
    arm_servo.start(0)
    arm_servo.ChangeDutyCycle(7.5)  # Adjust as per your servo's requirement
    time.sleep(1)
    arm_servo.ChangeDutyCycle(0)  # Stop the servo
    arm_servo.stop()

# Function to calculate distance from object (dummy implementation)
def calculate_distance(detection_box, frame_width, frame_height):
    ymin, xmin, ymax, xmax = detection_box
    box_width = (xmax - xmin) * frame_width
    box_height = (ymax - ymin) * frame_height
    distance = 1000 / (box_width * box_height)  # Arbitrary scaling
    return distance

# Object detection function
def detect_and_act():
    cap = cv2.VideoCapture(0)  # Open webcam
    target_object = "eraser"  # Target object name
    detection_threshold = 0.5  # Confidence threshold
    target_distance = 30  # Distance threshold to stop the car (in arbitrary units)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Prepare frame for model
        input_tensor = tf.convert_to_tensor(frame)
        input_tensor = input_tensor[tf.newaxis, ...]

        # Perform detection
        detections = model(input_tensor)

        # Extract details from detections
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}

        detection_boxes = detections['detection_boxes']
        detection_classes = detections['detection_classes'].astype(np.int32)
        detection_scores = detections['detection_scores']

        for i in range(num_detections):
            if detection_scores[i] > detection_threshold:  # Confidence check
                detected_object = "eraser" if detection_classes[i] == 1 else None
                
                if detected_object == target_object:
                    # Get detection box and calculate distance
                    box = detection_boxes[i]
                    frame_height, frame_width, _ = frame.shape
                    distance = calculate_distance(box, frame_width, frame_height)

                    # Move car closer to the object if far away
                    if distance > target_distance:
                        print(f"Moving towards {target_object}...")
                        move_forward()
                    else:
                        print(f"Stopping near {target_object}.")
                        stop_car()
                        time.sleep(1)
                        print("Activating robotic arm...")
                        pick_object()
                        break

        # Display the frame with bounding boxes (optional)
        cv2.imshow('Object Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    GPIO.cleanup()

# Run the function
detect_and_act()
