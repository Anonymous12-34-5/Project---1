import cv2
import mediapipe as mp
import pyautogui
import math
import time

# ================= SETTINGS =================
pyautogui.FAILSAFE = False
SMOOTHING = 1
CLICK_DISTANCE = 35

# Screen size
screen_w, screen_h = pyautogui.size()

# Camera
cap = cv2.VideoCapture(0)

# MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

prev_x, prev_y = 0, 0
last_click_time = 0

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    h, w, _ = img.shape

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            lm = hand.landmark

            # Index finger (8)
            ix, iy = int(lm[8].x * w), int(lm[8].y * h)

            # Thumb (4)
            tx, ty = int(lm[4].x * w), int(lm[4].y * h)

            # Map to screen
            screen_x = int(lm[8].x * screen_w)
            screen_y = int(lm[8].y * screen_h)

            # Smooth mouse
            curr_x = prev_x + (screen_x - prev_x) / SMOOTHING
            curr_y = prev_y + (screen_y - prev_y) / SMOOTHING

            pyautogui.moveTo(curr_x, curr_y)
            prev_x, prev_y = curr_x, curr_y

            # Pinch distance
            distance = math.hypot(tx - ix, ty - iy)

            # Click
            if distance < CLICK_DISTANCE:
                if time.time() - last_click_time > 0.5:
                    pyautogui.click()
                    last_click_time = time.time()

            # Draw
            cv2.circle(img, (ix, iy), 10, (0, 255, 0), -1)
            cv2.circle(img, (tx, ty), 10, (0, 0, 255), -1)
            mp_draw.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Hand Gesture Mouse", img)

    # ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
