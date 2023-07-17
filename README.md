# WEBCAM FACE RECOGNITION
This is a Python-based face recognition tool that identifies faces utilizing the webcam. The packages that are used in this tool are as follows: 
- `cv2`
- `tkinter`
- `face_recognition`
- `numpy`

# Files involved
- `headshot.py` - This script is utilized to take user photos. The program takes five images and saves them in a directory with the user's name inside the faces directory.
  This script takes the user's name as a command line argument. (Example: `python3 headshot.py John`)
- `recognition.py` - This script is used to run recognition based on the images of the user that are present inside the `faces` directory.
- `gui.py` - This script utilizes a GUI to run the `headshot.py` and `recognition.py` scripts.
