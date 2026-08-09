import cv2
from deepface import DeepFace
import time

# Open webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("❌ Camera could not be opened")
    exit()

# OpenCV's built-in face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

last_emotion = "Detecting..."
last_confidence = 0
last_analysis_time = 0

print("====================================")
print("   AI EMOTION DETECTION")
print("====================================")
print("Camera started.")
print("Look at the camera.")
print("Press Q to quit.")
print()

while True:

    ret, frame = camera.read()

    if not ret:
        print("❌ Could not read camera")
        break

    # Mirror effect
    frame = cv2.flip(frame, 1)

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80)
    )

    # If face found
    for (x, y, w, h) in faces:

        # Draw face rectangle
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        # Analyze only once every 1.5 seconds
        current_time = time.time()

        if current_time - last_analysis_time > 1.5:

            face = frame[y:y+h, x:x+w]

            try:
                print("🤖 Analyzing emotion...")

                result = DeepFace.analyze(
                    face,
                    actions=["emotion"],
                    detector_backend="skip",
                    enforce_detection=False
                )

                if isinstance(result, list):
                    result = result[0]

                last_emotion = result["dominant_emotion"]
                last_confidence = result["emotion"][last_emotion]

                print(
                    f"😊 Emotion: {last_emotion.upper()} "
                    f"({last_confidence:.1f}%)"
                )

            except Exception as e:
                print("❌ Emotion error:", e)

            last_analysis_time = current_time

        # Display emotion
        cv2.putText(
            frame,
            f"Emotion: {last_emotion.upper()}",
            (x, y - 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Confidence: {last_confidence:.1f}%",
            (x, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

    # No face
    if len(faces) == 0:

        cv2.putText(
            frame,
            "No face detected",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2
        )

    # Display camera
    cv2.imshow("AI Emotion Detection", frame)

    # Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()