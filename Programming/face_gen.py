import cv2
import os

# Получаем путь к текущему скрипту
path = os.path.dirname(os.path.abspath(__file__))

# Путь к файлу каскада Хаара (предполагаем, что он в той же директории)
cascade_path = os.path.join(path, "haarcascade_frontalface_default.xml")

# Проверяем, существует ли файл каскада
if not os.path.exists(cascade_path):
    print(f"Ошибка: Файл каскада не найден по пути: {cascade_path}")
    exit()

# Указываем, что мы будем искать лица по примитивам Хаара
detector = cv2.CascadeClassifier(cascade_path)

# Счётчик изображений
i = 0
# Расстояние от распознанного лица до рамки
offset = 50
# Запрашиваем номер пользователя
name = input('Введите номер пользователя: ')

# Создаем директорию dataSet, если она не существует
data_set_path = os.path.join(path, "dataSet")
if not os.path.exists(data_set_path):
    os.makedirs(data_set_path)

# Получаем доступ к камере
video = cv2.VideoCapture(0)

# Проверяем, успешно ли открылась камера
if not video.isOpened():
    print("Ошибка: Не удалось открыть камеру.")
    exit()

# Запускаем цикл
while True:
    # Берём видеопоток
    ret, im = video.read()

    # Если не удалось получить кадр, выходим из цикла
    if not ret:
        print("Ошибка: Не удалось получить кадр с камеры.")
        break

    # Переводим всё в ч/б для простоты
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # Настраиваем параметры распознавания и получаем лицо с камеры
    faces = detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100))
    # Обрабатываем лица
    for (x, y, w, h) in faces:
        # Увеличиваем счётчик кадров
        i = i + 1
        try:
            # Записываем файл на диск
            image_path = os.path.join(data_set_path, f"face-{name}.{i}.jpg")
            cv2.imwrite(image_path, gray[y - offset:y + h + offset, x - offset:x + w + offset])
            # Формируем размеры окна для вывода лица
            cv2.rectangle(im, (x - 50, y - 50), (x + w + 50, y + h + 50), (225, 0, 0), 2)
            # Показываем очередной кадр, который мы запомнили
            cv2.imshow('im', im[y - offset:y + h + offset, x - offset:x + w + offset])
            # Делаем паузу
            cv2.waitKey(100)
        except Exception as e:
            print(f"Ошибка при сохранении изображения: {e}")

    # Если у нас хватает кадров
    if i > 30:
        # Освобождаем камеру
        video.release()
        # Удалаяем все созданные окна
        cv2.destroyAllWindows()
        # Останавливаем цикл
        break

# Освобождаем камеру и закрываем окна (если цикл завершился до i > 30)
video.release()
cv2.destroyAllWindows()
