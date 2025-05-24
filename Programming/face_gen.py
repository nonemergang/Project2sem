import face_recognition
import cv2
import pickle
import os

# Find path of XML file containing Haar Cascade
cascPathface = os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
faceCascade = cv2.CascadeClassifier(cascPathface)

# Load known faces and embeddings
data = pickle.loads(open('face_enc', "rb").read())

# Load the input image
image_path = "path/to/your/image.jpg"  # Replace with your image path
image = cv2.imread(image_path)
if image is None:
    print("Error: Could not load image.")
    exit()

rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces using Haar Cascade
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(60, 60),
    flags=cv2.CASCADE_SCALE_IMAGE
)

# Compute facial embeddings for detected faces
encodings = face_recognition.face_encodings(rgb)
names = []

for encoding in encodings:
    matches = face_recognition.compare_faces(data["encodings"], encoding)
    name = "Unknown"

    if True in matches:
        matchedIdxs = [i for (i, b) in enumerate(matches) if b]
        counts = {}

        for i in matchedIdxs:
            name = data["names"][i]
            counts[name] = counts.get(name, 0) + 1

        name = max(counts, key=counts.get)
    names.append(name)

# Draw results
for ((x, y, w, h), name) in zip(faces, names):
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(image, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

cv2.imshow("Frame", image)
cv2.waitKey(0)
cv2.destroyAllWindows()