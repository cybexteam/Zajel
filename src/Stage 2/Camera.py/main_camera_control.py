
import cv2
import numpy as np
import serial
import time
import math

# Initialize serial communication with ESP32
# Make sure to set the correct serial port and baudrate
SERIAL_PORT = '/dev/ttyUSB0'  # Change to your port
BAUD_RATE = 115200
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
time.sleep(2)

# HSV thresholds for color detection (can be adjusted)
HSV_RED = (np.array([170, 100, 80]), np.array([180, 255, 255]))
HSV_GREEN = (np.array([50, 50, 50]), np.array([80, 255, 255]))
HSV_MAGENTA = (np.array([140, 50, 50]), np.array([170, 255, 255]))
HSV_BLACK = (np.array([0, 0, 0]), np.array([180, 255, 50]))

# PID constants
KP = 0.4
KI = 0.0
KD = 0.2

# Steering camera offset (adjusted dynamically)
# The steering servo affects camera angle. This variable should be updated with the angle sent to the ESP32.
CAMERA_SERVO_ANGLE_OFFSET = 0  # Replace with live value if synchronized in real-time

# Counters for laps and turns
corner_counter = 0
lap_counter = 0

# To determine if robot is in parking phase
parking_mode = False

# PID state variables
prev_error = 0
integral = 0

# Start video capture (OAK-D or compatible camera)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

def detect_color_mask(frame, lower, upper):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, lower, upper)

def calculate_pid(error):
    global prev_error, integral
    integral += error
    derivative = error - prev_error
    prev_error = error
    return KP * error + KI * integral + KD * derivative

def send_command(cmd):
    ser.write((cmd + "\n").encode())

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize and crop for ROI processing
    roi = frame[60:250, :]
    lower_roi = frame[440:480, :]  # Bottom area for black line detection

    # Detect obstacles and path elements
    red_mask = detect_color_mask(roi, *HSV_RED)
    green_mask = detect_color_mask(roi, *HSV_GREEN)
    magenta_mask = detect_color_mask(roi, *HSV_MAGENTA)
    black_mask = detect_color_mask(lower_roi, *HSV_BLACK)

    # Count black pixels to detect path corners (black boundary)
    black_ratio = np.sum(black_mask > 0) / black_mask.size

    if black_ratio > 0.4:
        corner_counter += 1
        print(f"Corner detected: {corner_counter}")
        time.sleep(0.3)  # Debounce

        if corner_counter % 4 == 0:
            lap_counter += 1
            print(f"Lap: {lap_counter}")

        if lap_counter >= 3:
            parking_mode = True

    # Calculate object centroids for red and green
    contours_r, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_g, _ = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    target_x = None
    if contours_r:
        c = max(contours_r, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        target_x = x + w // 2
    elif contours_g:
        c = max(contours_g, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        target_x = x + w // 2

    if target_x is not None and not parking_mode:
        # Calculate deviation from center (320 is midline)
        error = 320 - target_x

        # Adjust error using camera steering angle offset
        # This compensates for the fact that the camera is physically rotated
        # according to the steering servo. This offset must be updated externally.
        error -= CAMERA_SERVO_ANGLE_OFFSET

        correction = calculate_pid(error)

        # Based on correction value, send commands to ESP
        if correction > 20:
            send_command("LEFT")
        elif correction < -20:
            send_command("RIGHT")
        else:
            send_command("FORWARD")

    elif parking_mode:
        # Search for magenta markers to trigger parking
        contours_m, _ = cv2.findContours(magenta_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours_m:
            send_command("PARK")
            print("Parking triggered")
            break
        else:
            send_command("FORWARD")

    else:
        send_command("FORWARD")

    # Display (for debugging)
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
ser.close()
