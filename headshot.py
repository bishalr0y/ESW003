import cv2
import sys
import os

def click_pics(directory_name):
    faces_folder = "faces"
    if not os.path.exists(faces_folder):
        os.makedirs(faces_folder)

    directory_path = os.path.join(faces_folder, directory_name)
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    key = cv2.waitKey(1)
    webcam = cv2.VideoCapture(0)
    count = 1

    while count <= 5:
        try:
            check, frame = webcam.read()
            print(check)  # prints true as long as the webcam is running
            cv2.imshow("Capturing", frame)
            key = cv2.waitKey(1)
            if key == ord('s'):
                file_name = os.path.join(directory_path, str(count) + '.jpg')
                cv2.imwrite(filename=file_name, img=frame)
                print("Image " + str(count) + " saved!")
                count += 1

            elif key == ord('q'):
                print("Turning off camera.")
                webcam.release()
                print("Camera off.")
                print("Program ended.")
                cv2.destroyAllWindows()
                break

        except KeyboardInterrupt:
            print("Turning off camera.")
            webcam.release()
            print("Camera off.")
            print("Program ended.")
            cv2.destroyAllWindows()
            break

    webcam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide the directory name as a command line argument.")
    else:
        directory_name = sys.argv[1]
        click_pics(directory_name)
import cv2
import sys
import os

def click_pics(directory_name):
    faces_folder = "faces"
    if not os.path.exists(faces_folder):
        os.makedirs(faces_folder)

    directory_path = os.path.join(faces_folder, directory_name)
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    key = cv2.waitKey(1)
    webcam = cv2.VideoCapture(0)
    count = 1

    while count <= 5:
        try:
            check, frame = webcam.read()
            print(check)  # prints true as long as the webcam is running
            cv2.imshow("Capturing", frame)
            key = cv2.waitKey(1)
            if key == ord('s'):
                file_name = os.path.join(directory_path, str(count) + '.jpg')
                cv2.imwrite(filename=file_name, img=frame)
                print("Image " + str(count) + " saved!")
                count += 1

            elif key == ord('q'):
                print("Turning off camera.")
                webcam.release()
                print("Camera off.")
                print("Program ended.")
                cv2.destroyAllWindows()
                break

        except KeyboardInterrupt:
            print("Turning off camera.")
            webcam.release()
            print("Camera off.")
            print("Program ended.")
            cv2.destroyAllWindows()
            break

    webcam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide the directory name as a command line argument.")
    else:
        directory_name = sys.argv[1]
        click_pics(directory_name)
