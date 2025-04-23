import cv2
import torch

# Загрузка модели YOLOv8
model = torch.hub.load('ultralytics/yolov8', 'yolov8n')  # Или другая версия

# Захват видеопотока
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Детекция объектов
    results = model(frame)

    # Обработка результатов
    for *xyxy, conf, cls in results.xyxy[0]:
        if conf > 0.5 and cls == 0:  # 0 - класс "person" (человек)
            # Рисование прямоугольника вокруг обнаруженного человека
            x1, y1, x2, y2 = map(int, xyxy)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Подсчет людей (простой подсчет обнаружений, без трэкинга)
    num_people = len(results.xyxy[0])

    # Вывод количества людей на кадре
    cv2.putText(frame, f"People: {num_people}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Отображение кадра
    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
