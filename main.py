import subprocess
import time
import cv2
import mediapipe as mp
import pyautogui

# Using mediapipe to set up hand recognition
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)


def distance(a, b):
    return abs(a - b)


def is_touching(landmark_a, landmark_b, threshold):
    if distance(landmark_a, landmark_b) <= threshold:
        return True
    else:
        return False


class Point3D(object):
    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z


def mirror(landmark):
    res = []
    for l in landmark:
        res.append(Point3D(-l.x, l.y, l.z))
    return res


# Global variables
state = "unknown"
state_distance = 0
vlc_running = False
start_time = time.time()


def position(p):
    global state
    global start_time

    if state == "unknown":
        start_time = time.time()
        state = p
        print("unknown -> ", p)
    elif state != "hold" and (time.time() - start_time) > 0.3:
        action(p)
        state = "hold"
        print(time.time(), " : known -> hold")


def position_volume(p, d):
    global state
    global state_distance
    global start_time

    if state == "unknown":
        start_time = time.time()
        state = p
        state_distance = d
        print("unknown -> ", p)
    elif state == p and (time.time() - start_time) > 0.3:
        if d - state_distance > 0.10 * state_distance:
            action("volume_up")
            state_distance = d
        elif state_distance - d > 0.10 * state_distance:
            action("volume_down")
            state_distance = d


def send_to_vlc(*key):
    window_title = "debussy_arabesque_num_1.mp4 - Lecteur multimédia VLC"

    # Searches for all VLC windows open (to fix the 'stop' issue)
    titles = pyautogui.getAllTitles()
    for t in titles:
        if 'Lecteur multimédia VLC' in t:
            window_title = t

    windows = pyautogui.getWindowsWithTitle(window_title)

    if windows:
        window = windows[0]
        window.activate()
        pyautogui.hotkey(*key)


def action(p):
    global vlc_running

    if p == "thumb_up":
        if not vlc_running:
            launch_vlc()
        send_to_vlc('space')
    elif p == "open_hand":
        send_to_vlc('s')
    elif p == "zero_hand":
        send_to_vlc('m')
    elif p == "victory_hand":
        send_to_vlc('right')
    elif p == "fist_hand":
        send_to_vlc('ctrl', 'q')
        vlc_running = False
    elif p == "fists_hands":
        send_to_vlc('ctrl', 'q')
        exit(0)
    elif p == "victories_hands":
        send_to_vlc('left')
    elif p == "square_hands":
        send_to_vlc('f')
    elif p == "volume_up":
        send_to_vlc('ctrl', 'up')
    elif p == "volume_down":
        send_to_vlc('ctrl', 'down')


def thumb_up(handLandmarks):
    return handLandmarks[4].y < handLandmarks[3].y \
        and handLandmarks[8].x > handLandmarks[6].x \
        and handLandmarks[12].x > handLandmarks[10].x \
        and handLandmarks[16].x > handLandmarks[14].x \
        and handLandmarks[20].x > handLandmarks[18].x \
        and handLandmarks[17].x < handLandmarks[0].x


def open_hand(handLandmarks):
    return handLandmarks[4].x < handLandmarks[3].x \
        and handLandmarks[8].y < handLandmarks[6].y \
        and handLandmarks[12].y < handLandmarks[10].y \
        and handLandmarks[16].y < handLandmarks[14].y \
        and handLandmarks[20].y < handLandmarks[18].y \
        and handLandmarks[17].x > handLandmarks[0].x


def zero_hand(handLandmarks):
    t = distance(handLandmarks[0].y, handLandmarks[5].y) / 5
    return is_touching(handLandmarks[4].x, handLandmarks[8].x, t) \
        and is_touching(handLandmarks[4].y, handLandmarks[8].y, t) \
        and handLandmarks[6].y < handLandmarks[8].y \
        and handLandmarks[12].y < handLandmarks[10].y \
        and handLandmarks[16].y < handLandmarks[14].y \
        and handLandmarks[20].y < handLandmarks[18].y \
        and handLandmarks[17].x > handLandmarks[0].x


def victory_hand(handLandmarks):
    return handLandmarks[4].x > handLandmarks[3].x \
        and handLandmarks[8].y < handLandmarks[6].y \
        and handLandmarks[12].y < handLandmarks[10].y \
        and handLandmarks[16].y > handLandmarks[14].y \
        and handLandmarks[20].y > handLandmarks[18].y \
        and handLandmarks[17].x > handLandmarks[0].x


def fist_hand(handLandmarks):
    t = distance(handLandmarks[0].y, handLandmarks[5].y) / 5
    return is_touching(handLandmarks[4].x, handLandmarks[10].x, t) \
        and is_touching(handLandmarks[4].y, handLandmarks[10].y, t) \
        and handLandmarks[8].y > handLandmarks[6].y \
        and handLandmarks[12].y > handLandmarks[10].y \
        and handLandmarks[16].y > handLandmarks[14].y \
        and handLandmarks[20].y > handLandmarks[18].y \
        and handLandmarks[17].x > handLandmarks[0].x


def square_hands(rightLandmarks, leftLandmarks):
    t = distance(rightLandmarks[4].y, rightLandmarks[3].y)
    return rightLandmarks[4].y < rightLandmarks[3].y \
        and rightLandmarks[17].x < rightLandmarks[0].x \
        and leftLandmarks[4].y > leftLandmarks[3].y \
        and leftLandmarks[17].x > leftLandmarks[0].x \
        and is_touching(rightLandmarks[4].y, leftLandmarks[8].y, t) \
        and is_touching(rightLandmarks[4].x, leftLandmarks[8].x, t) \
        and is_touching(leftLandmarks[4].y, rightLandmarks[8].y, t) \
        and is_touching(leftLandmarks[4].x, rightLandmarks[8].x, t)


def index_hand(handLandmarks):
    return handLandmarks[4].x > handLandmarks[3].x \
        and handLandmarks[8].y < handLandmarks[6].y \
        and handLandmarks[12].y > handLandmarks[10].y \
        and handLandmarks[16].y > handLandmarks[14].y \
        and handLandmarks[20].y > handLandmarks[18].y \
        and handLandmarks[17].x > handLandmarks[0].x


def event_loop():
    global state

    prev_time = 0
    while cap.isOpened():
        success, image = cap.read()
        image = cv2.flip(image, 1)

        if not success:
            print("Ignoring empty camera frame.")
            continue

        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if results.multi_hand_landmarks:

            if len(results.multi_hand_landmarks) == 2:

                left_landmarks = results.multi_hand_landmarks[0]
                right_landmarks = results.multi_hand_landmarks[1]

                if fist_hand(right_landmarks.landmark) and fist_hand(mirror(left_landmarks.landmark)):
                    position("fists_hands")
                elif victory_hand(right_landmarks.landmark) and victory_hand(mirror(left_landmarks.landmark)):
                    position("victories_hands")
                elif index_hand(right_landmarks.landmark) and index_hand(mirror(left_landmarks.landmark)):
                    d = distance(right_landmarks.landmark[8].x, left_landmarks.landmark[8].x)
                    position_volume("index_hands", d)
                elif square_hands(right_landmarks.landmark, left_landmarks.landmark):
                    position("square_hands")
                else:
                    state = "unknown"
                    print(" -> unknown")

                # Draw hand landmarks
                mp_drawing.draw_landmarks(image, left_landmarks, mp_hands.HAND_CONNECTIONS)
                mp_drawing.draw_landmarks(image, right_landmarks, mp_hands.HAND_CONNECTIONS)

            elif len(results.multi_hand_landmarks) == 1:

                # To find the first hand that appears in the frame
                hand_landmarks = results.multi_hand_landmarks[0]

                # To find out which hand it is
                hand_label = results.multi_handedness[0].classification[0].label

                # Ranked from most to least restrictive
                if hand_label == 'Right':
                    if zero_hand(hand_landmarks.landmark):
                        position("zero_hand")
                    elif thumb_up(hand_landmarks.landmark):
                        position("thumb_up")
                    elif fist_hand(hand_landmarks.landmark):
                        position("fist_hand")
                    elif victory_hand(hand_landmarks.landmark):
                        position("victory_hand")
                    elif open_hand(hand_landmarks.landmark):
                        position("open_hand")
                    else:
                        state = "unknown"
                        print(" -> unknown")

                # Draw hand landmarks
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        else:
            state = "unknown"
            print(" no hand -> unknown")

        # Fps
        cur_time = time.time()
        fps = int(1 / (cur_time - prev_time))
        prev_time = cur_time
        text = "fps : " + str(fps)

        cv2.putText(image, text, (40, 60), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 2)

        # Display image
        cv2.imshow('AirKey - Camera', image)

        # Kill
        if cv2.waitKey(1) == 27:
            break


# Webcam input:
cap = cv2.VideoCapture(0)

# Help window
help_image = cv2.imread('sign_dictionnary.png', cv2.IMREAD_UNCHANGED)

scale_percent = 40  # percent of original size
width = int(help_image.shape[1] * scale_percent / 100)
height = int(help_image.shape[0] * scale_percent / 100)
dim = (width, height)
resized = cv2.resize(help_image, dim, interpolation=cv2.INTER_AREA)

cv2.imshow("AirKey - Help", resized)


def launch_vlc():
    global vlc_running
    # Using the subprocess module to open VLC
    vlc_path = r"C:\Program Files\VideoLAN\VLC\vlc.exe"
    media_file = r'C:\Users\Albane\Pictures\Perso\debussy_arabesque_num_1.mp4'  # To modify with your own path
    subprocess.Popen([vlc_path, media_file])
    vlc_running = True


def main():
    launch_vlc()
    event_loop()
    cap.release()
    cv2.destroyAllWindows()


main()
