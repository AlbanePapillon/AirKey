import time
import cv2
import mediapipe as mp
import vlc


path = r'C:\Users\Albane\Pictures\Perso\debussy_arabesque_num_1.mp4'

# hand recognition
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)

# VLC player
instance = vlc.Instance()
player = instance.media_player_new()

media = instance.media_new(path)
player.set_media(media)

# time management
start_init = False
prev_time = 0


def distance(a, b):
    return abs(a - b)


def fullscreen(hand_label):
    epsilon = 0.5

    left_thumb = 0
    left_index = 0
    right_thumb = 0
    right_index = 0

    if hand_label == "Left":
        left_thumb = handLandmarks[4][0]
        left_index = handLandmarks[8][1]

    if hand_label == "Right":
        right_thumb = handLandmarks[4][0]
        right_index = handLandmarks[8][1]

    if distance(left_thumb, right_index) < epsilon and distance(right_thumb, left_index) < epsilon:
        print("fullscreen")
        player.toggle_fullscreen()


def play_pause(is_playing):
    if (not is_playing) and handLandmarks[4][0] > handLandmarks[3][0]:
        player.play
        is_playing = True
    elif is_playing and handLandmarks[4][0] > handLandmarks[3][0]:
        player.pause


def actions_vlc(cnt, prev=-1):
    if prev != cnt:
        if cnt == 0:
            print(0)

        elif cnt == 1:
            print(1)
            player.play()

        elif cnt == 2:
            print(2)

        elif cnt == 3:
            print(3)
            player.pause()

        elif cnt == 4:
            print(4)

        elif cnt == 5:
            print(5)
            player.stop()
            instance.release()

        elif cnt == 6:
            print(6)

        elif cnt == 7:
            print(7)

        elif cnt == 8:
            print(8)

        elif cnt == 9:
            print(9)

        elif cnt == 10:
            print(10)

        prev = cnt


# For webcam input:
cap = cv2.VideoCapture(0)

while cap.isOpened():
    end_time = time.time()

    success, image = cap.read()
    image = cv2.flip(image, 1)

    if not success:
        print("Ignoring empty camera frame.")
        continue

    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Initially set finger count to 0 for each cap
    fingerCount = 0

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:
            # Get hand index to check label (left or right)
            handIndex = results.multi_hand_landmarks.index(hand_landmarks)
            handLabel = results.multi_handedness[handIndex].classification[0].label

            # Fill list with x and y positions of each landmark
            handLandmarks = []
            for landmarks in hand_landmarks.landmark:
                handLandmarks.append([landmarks.x, landmarks.y])

            play_pause(False)

            fullscreen(handLabel)

            # Thumb: TIP x position must be greater or lower than IP x position,
            #   depending on hand label.
            if handLabel == "Left" and handLandmarks[4][0] > handLandmarks[3][0]:
                fingerCount += 1

            elif handLabel == "Right" and handLandmarks[4][0] < handLandmarks[3][0]:
                fingerCount += 1

            # Other fingers: TIP y position must be lower than PIP y position,
            #   as image origin is in the upper left corner.
            if handLandmarks[8][1] < handLandmarks[6][1]:  # Index finger
                fingerCount += 1

            if handLandmarks[12][1] < handLandmarks[10][1]:  # Middle finger
                fingerCount += 1

            if handLandmarks[16][1] < handLandmarks[14][1]:  # Ring finger
                fingerCount += 1

            if handLandmarks[20][1] < handLandmarks[18][1]:  # Pinky
                fingerCount += 1

            # Draw hand landmarks
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    if not start_init:
        start_time = time.time()
        start_init = True

    elif (end_time - start_time) > 1:
        actions_vlc(fingerCount, False)
        start_init = False

    # calcul fps
    cur_time = time.time()
    fps = int(1 / (cur_time - prev_time))
    prev_time = cur_time
    # print(fps)

    # Display image
    cv2.imshow('MediaPipe Hands', image)

    # Echap
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
