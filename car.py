import cv2

# Load the video file and create a foreground detector
the_video = cv2.VideoCapture('video/traffic.avi')
object_detector = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=1000)

# Define minimum size threshold for the detected objects
min_size = 1000

# Read frames from the video and detect the foreground objects
for i in range(150):
    ret, frame = the_video.read()
    the_object = object_detector.apply(frame)

# Display the original video frame and the detected object
cv2.imshow('Video Frame', frame)
cv2.imshow('The Object', the_object)

# Remove noise from the detected object
structure = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
noise_free_object = cv2.morphologyEx(the_object, cv2.MORPH_CLOSE, structure)
cv2.imshow('Object After Removing Noise', noise_free_object)

# Analyze the blobs in the image and draw bounding boxes around the cars
contours, hierarchy = cv2.findContours(noise_free_object, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
bounding_boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > min_size]
detected_car = frame.copy()
for box in bounding_boxes:
    detected_car = cv2.rectangle(detected_car, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (0, 255, 0), 2)

# Count the number of cars and display the result
number_of_cars = len(bounding_boxes)
detected_car = cv2.putText(detected_car, f"Number of Cars: {number_of_cars}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
cv2.imshow('Detected Cars', detected_car)

# Create a video player to display the video with the detected cars
video_player = cv2.VideoCapture('video/traffic.avi')
while video_player.isOpened():
    ret, frame = video_player.read()
    if not ret:
        break
    the_object = object_detector.apply(frame) 
    noise_free_object = cv2.morphologyEx(the_object, cv2.MORPH_CLOSE, structure) 
    contours, hierarchy = cv2.findContours(noise_free_object, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > min_size]
    detected_car = frame.copy()
    for box in bounding_boxes:
        detected_car = cv2.rectangle(detected_car, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (0, 255, 0), 2)
    number_of_cars = len(bounding_boxes) 
    detected_car = cv2.putText(detected_car, f"Number of Cars: {number_of_cars}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2) 
    cv2.imshow('Detected Cars', detected_car)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
video_player.release()
cv2.destroyAllWindows()