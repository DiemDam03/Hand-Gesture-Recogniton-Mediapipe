
import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
from mediapipe.framework.formats import landmark_pb2

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Label box parameters
label_text_color = (255, 0, 0)  # white
label_font_size = 1
label_thickness = 2

recognition_result_list = []

def save_result(result: GestureRecognizerResult, output_image, timestamp_ms: int):
    recognition_result_list.append(result)
    print(result.gestures)

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path='C:/Users/buiph/Desktop/New PyAI/GestureRecognition/gesture_recognizer.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    min_hand_detection_confidence = 0.7,
    min_hand_presence_confidence = 0.7,
    min_tracking_confidence = 0.7,
    result_callback=save_result)

cap = cv2.VideoCapture(0)

timestamp = 0
with GestureRecognizer.create_from_options(options) as recognizer:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        timestamp += 1

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        recognizer.recognize_async(mp_image, timestamp)

        if recognition_result_list:
            for hand_index, hand_landmarks in enumerate(
                recognition_result_list[0].hand_landmarks):
                # Calculate the bounding box of the hand
                x_min = min([landmark.x for landmark in hand_landmarks])
                y_min = min([landmark.y for landmark in hand_landmarks])
                y_max = max([landmark.y for landmark in hand_landmarks]) 

                # Convert normalized coordinates to pixel values
                frame_height, frame_width = image.shape[:2]
                x_min_px = int(x_min * frame_width)
                y_min_px = int(y_min * frame_height)
                y_max_px = int(y_max * frame_height)

                # Get gesture classification results
                if recognition_result_list[0].gestures:
                    gesture = recognition_result_list[0].gestures[hand_index]
                    category_name = gesture[0].category_name
                    score = round(gesture[0].score, 2)
                    result_text = f'{category_name} '

                    # Compute text size
                    text_size = \
                    cv2.getTextSize(result_text, cv2.FONT_HERSHEY_DUPLEX, label_font_size,
                            label_thickness)[0]
                    text_width, text_height = text_size

                    # Calculate text position (above the hand)
                    text_x = x_min_px
                    text_y = y_min_px - 10  # Adjust this value as needed

                    # Make sure the text is within the frame boundaries
                    if text_y < 0:
                        text_y = y_max_px + text_height

                    # Draw the text
                    cv2.putText(image, result_text, (text_x, text_y),
                        cv2.FONT_HERSHEY_DUPLEX, label_font_size,
                        label_text_color, label_thickness, cv2.LINE_AA)
                    
                # Draw hand landmarks on the frame
                hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                hand_landmarks_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y,
                                                    z=landmark.z) for landmark in
                                                    hand_landmarks])
                mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks_proto,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
            recognition_result_list.clear()

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
