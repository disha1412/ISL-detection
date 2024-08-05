import os
import cv2

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 3
dataset_size = 100

# Try different camera indices
camera_indices = [0,1,2]

cap = None
for index in camera_indices:
    cap = cv2.VideoCapture(index)
    if cap is not None and cap.isOpened():
        break

if cap is None or not cap.isOpened():
    print("Error: Could not open any camera.")
else:
    for j in range(number_of_classes):
        class_dir = os.path.join(DATA_DIR, str(j))
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)

        print(f'Collecting data for class {j}')

        # Wait for user to be ready
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture image.")
                break
            cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
            cv2.imshow('frame', frame)
            if cv2.waitKey(25) == ord('q'):
                break

        counter = 0
        while counter < dataset_size:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture image.")
                break
            cv2.imshow('frame', frame)
            cv2.waitKey(25)
            cv2.imwrite(os.path.join(class_dir, f'{counter}.jpg'), frame)
            counter += 1

    cap.release()
    cv2.destroyAllWindows()
