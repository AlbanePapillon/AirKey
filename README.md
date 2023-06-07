# AirKey

This project utilizes hand recognition and gesture control to interact with a media player (VLC). By performing specific hand gestures in front of a webcam, you can control various functions of the media player.

## Requirements

- Python 3.x
- OpenCV
- Mediapipe
- PyAutoGUI
- VLC media player

## Installation

1. Clone the repository or download the source code files.
2. Install the required Python packages using pip: `pip install opencv-python mediapipe pyautogui`
3. Install VLC media player from the official website: https://www.videolan.org/vlc/index.html

## Usage

1. Connect a webcam to your computer.
2. Run the `main.py` script: `python main.py`
3. The webcam feed will open along with a help window showing the supported hand gestures.
4. Perform the specified hand gestures in front of the webcam to control the media player functions.
5. The supported hand gestures and their corresponding actions are as follows:

- Thumb Up: Play/Pause
- Open Hand: Stop
- Zero Hand: Mute
- Victory Hand: Skip Forward
- Fist Hand: Quit VLC
- Fists Hands: Quit VLC and Exit Program
- Victories Hands: Skip Backward
- Square Hands: Fullscreen Toggle
- Index Hands: Volume Control (Pinch to Increase/Decrease)

6. If needed, press the 'Esc' key to exit the program.

## Customization

- To use a different media file, modify the `media_file` variable in the `launch_vlc()` function.

## Acknowledgments

This project utilizes the following libraries and tools:

- OpenCV: https://opencv.org/
- Mediapipe: https://mediapipe.dev/
- PyAutoGUI: https://pyautogui.readthedocs.io/
- VLC media player: https://www.videolan.org/vlc/index.html

## Contributors
- https://github.com/AlbanePapillon
- https://github.com/GDutheil
- https://github.com/TheoPinardin

