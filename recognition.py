import face_recognition
import os
import sys
import cv2
import numpy as np
import math
import re


# Helper
def face_confidence(face_distance, face_match_threshold=0.5):
    range = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (range * 2.0)

    if face_distance > face_match_threshold:
        return str(round(linear_val * 100, 2)) + '%'
    else:
        value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        return str(round(value, 2)) + '%'


class FaceRecognition:
    face_locations = []
    face_encodings = []
    face_names = []
    known_face_encodings = []
    known_face_names = []
    process_current_frame = True

    def __init__(self):
        self.encode_faces()

    def encode_faces(self):
        for person_dir in os.listdir('faces'):
            person_path = os.path.join('faces', person_dir)
            if os.path.isdir(person_path):
                person_encodings = []
                for image_name in os.listdir(person_path):
                    image_path = os.path.join(person_path, image_name)
                    try:
                        face_image = face_recognition.load_image_file(image_path)
                        face_encoding = face_recognition.face_encodings(face_image, num_jitters=1)[0]
                        person_encodings.append(face_encoding)
                    except IndexError:
                        print(f"No face found in {image_path}. Skipping...")
                if person_encodings:
                    self.known_face_encodings.append(person_encodings)
                    self.known_face_names.append(person_dir)
        print(self.known_face_names)

    def run_recognition(self):
        video_capture = cv2.VideoCapture(0)

        if not video_capture.isOpened():
            sys.exit('Video source not found...')

        while True:
            ret, frame = video_capture.read()

            # Only process every other frame of video to save time
            if self.process_current_frame:
                # Resize frame of video to 1/4 size for faster face recognition processing
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

                # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                # Find all the faces and face encodings in the current frame of video
                self.face_locations = face_recognition.face_locations(rgb_small_frame)
                self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)

                self.face_names = []
                for face_encoding in self.face_encodings:
                    # See if the face is a match for any of the known persons
                    best_person_index = -1
                    best_match_distance = float('inf')

                    for i, person_encodings in enumerate(self.known_face_encodings):
                        matches = face_recognition.compare_faces(person_encodings, face_encoding)
                        avg_distance = np.average(face_recognition.face_distance(person_encodings, face_encoding))

                        if all(matches) and avg_distance < best_match_distance:
                            best_person_index = i
                            best_match_distance = avg_distance

                    if best_person_index != -1:
                        name = self.known_face_names[best_person_index]
                        confidence = face_confidence(best_match_distance)

                        if float(confidence.split('%')[0]) < 55:
                            name = "Unknown"
                            confidence = '???'
                        elif float(confidence.split('%')[0]) < 65:
                            name = "Not sure"
                            confidence = 'Improve the lighting'
                        else:
                            pass

                        self.face_names.append(f'{name} ({confidence})')

            self.process_current_frame = not self.process_current_frame

            # Display the results
            for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Create the frame with the name
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

            # Display the resulting image
            small_frame = cv2.resize(frame, (0, 0), fx=0.6, fy=0.6)

            cv2.imshow('Face Recognition', small_frame)

            # Hit 'q' on the keyboard to quit!
            if cv2.waitKey(1) == ord('q'):
                break

        # Release handle to the webcam
        video_capture.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    fr = FaceRecognition()
    fr.run_recognition()
import face_recognition
import os
import sys
import cv2
import numpy as np
import math
import re


# Helper
def face_confidence(face_distance, face_match_threshold=0.5):
    range = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (range * 2.0)

    if face_distance > face_match_threshold:
        return str(round(linear_val * 100, 2)) + '%'
    else:
        value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        return str(round(value, 2)) + '%'


class FaceRecognition:
    face_locations = []
    face_encodings = []
    face_names = []
    known_face_encodings = []
    known_face_names = []
    process_current_frame = True

    def __init__(self):
        self.encode_faces()

    def encode_faces(self):
        for person_dir in os.listdir('faces'):
            person_path = os.path.join('faces', person_dir)
            if os.path.isdir(person_path):
                person_encodings = []
                for image_name in os.listdir(person_path):
                    image_path = os.path.join(person_path, image_name)
                    try:
                        face_image = face_recognition.load_image_file(image_path)
                        face_encoding = face_recognition.face_encodings(face_image, num_jitters=1)[0]
                        person_encodings.append(face_encoding)
                    except IndexError:
                        print(f"No face found in {image_path}. Skipping...")
                if person_encodings:
                    self.known_face_encodings.append(person_encodings)
                    self.known_face_names.append(person_dir)
        print(self.known_face_names)

    def run_recognition(self):
        video_capture = cv2.VideoCapture(0)

        if not video_capture.isOpened():
            sys.exit('Video source not found...')

        while True:
            ret, frame = video_capture.read()

            # Only process every other frame of video to save time
            if self.process_current_frame:
                # Resize frame of video to 1/4 size for faster face recognition processing
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

                # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                # Find all the faces and face encodings in the current frame of video
                self.face_locations = face_recognition.face_locations(rgb_small_frame)
                self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)

                self.face_names = []
                for face_encoding in self.face_encodings:
                    # See if the face is a match for any of the known persons
                    best_person_index = -1
                    best_match_distance = float('inf')

                    for i, person_encodings in enumerate(self.known_face_encodings):
                        matches = face_recognition.compare_faces(person_encodings, face_encoding)
                        avg_distance = np.average(face_recognition.face_distance(person_encodings, face_encoding))

                        if all(matches) and avg_distance < best_match_distance:
                            best_person_index = i
                            best_match_distance = avg_distance

                    if best_person_index != -1:
                        name = self.known_face_names[best_person_index]
                        confidence = face_confidence(best_match_distance)

                        if float(confidence.split('%')[0]) < 55:
                            name = "Unknown"
                            confidence = '???'
                        elif float(confidence.split('%')[0]) < 65:
                            name = "Not sure"
                            confidence = 'Improve the lighting'
                        else:
                            pass

                        self.face_names.append(f'{name} ({confidence})')

            self.process_current_frame = not self.process_current_frame

            # Display the results
            for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Create the frame with the name
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

            # Display the resulting image
            small_frame = cv2.resize(frame, (0, 0), fx=0.6, fy=0.6)

            cv2.imshow('Face Recognition', small_frame)

            # Hit 'q' on the keyboard to quit!
            if cv2.waitKey(1) == ord('q'):
                break

        # Release handle to the webcam
        video_capture.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    fr = FaceRecognition()
    fr.run_recognition()
