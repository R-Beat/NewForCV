import cv2
from ultralytics import YOLO

# Load your trained model
model = YOLO(r"C:\Users\Asus\Desktop\GUI Research Code\runs\detect\train4\weights\best.pt")  # change to your model path

# Open the camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # change index if needed for DSLR

if not cap.isOpened():
    print("Error: Could not open camera.")
else:
    ret, frame = cap.read()  # Capture one frame
    if ret:
        # Run YOLO prediction
        results = model(frame)

        # Plot results on the image
        annotated_frame = results[0].plot()

        # Show the annotated frame
        cv2.imshow("Detected Objects", annotated_frame)
        cv2.waitKey(0)  # Wait for a key press
    else:
        print("Error: Failed to capture frame.")

# Release resources
cap.release()
cv2.destroyAllWindows()

#we could change the computer vision model, just change the path