import cv2
import pytesseract
from ultralytics import YOLO
from pathlib import Path
import shutil

# --- НАСТРОЙКИ ---
# Путь к Tesseract (ваш, проверенный)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Путь к обученной модели
MODEL_PATH = r"C:\DocumentCV\runs\detect\train\weights\best.pt"

# Папки
INPUT_DIR = Path("incoming")   # сюда кладём документы для обработки
OUTPUT_DIR = Path("output")    # сюда скрипт сложит результаты

# Создаём папки, если их нет
INPUT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Загружаем модель ОДИН РАЗ при запуске
print("Загружаю модель...")
model = YOLO(MODEL_PATH)
print("Модель загружена.\n")


def process_document(image_path: Path):
    """Обрабатывает один документ и создаёт папку с результатами."""
    doc_name = image_path.stem
    doc_dir = OUTPUT_DIR / doc_name
    doc_dir.mkdir(parents=True, exist_ok=True)

    # 1. Копируем исходный файл
    shutil.copy2(image_path, doc_dir / image_path.name)

    # 2. Читаем картинку
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"  [!] Не удалось прочитать: {image_path.name}")
        return

    # 3. Детекция через YOLO
    results = model(str(image_path), verbose=False)
    boxes = results[0].boxes

    box_lines = []
    found_photo = False
    found_mrz = False

    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            # Записываем в файл боксов
            box_lines.append(f"{cls_name} {conf:.3f} {x1} {y1} {x2} {y2}")

            # Вырезаем зону (с небольшим отступом на всякий случай)
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            if cls_name == "photo":
                cv2.imwrite(str(doc_dir / "face.jpg"), crop)
                found_photo = True

            elif cls_name == "mrz":
                cv2.imwrite(str(doc_dir / "mrz.jpg"), crop)

                # 4. OCR по MRZ
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                # Бинаризация для лучшего распознавания
                _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                mrz_text = pytesseract.image_to_string(
                    thresh,
                    lang="eng",
                    config="--psm 6"
                )
                with open(doc_dir / "mrz.txt", "w", encoding="utf-8") as f:
                    f.write(mrz_text.strip())

                found_mrz = True

    # 5. Сохраняем файл с боксами
    with open(doc_dir / "boxes.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(box_lines))

    # 6. Отчёт
    status = []
    status.append("photo=OK" if found_photo else "photo=НЕТ")
    status.append("mrz=OK" if found_mrz else "mrz=НЕТ")
    print(f"  {image_path.name}: {', '.join(status)}")


def main():
    # Ищем все картинки в папке incoming
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    images = []
    for ext in extensions:
        images.extend(INPUT_DIR.glob(ext))

    if not images:
        print(f"В папке '{INPUT_DIR}' нет картинок.")
        print("Положите туда файлы документов и запустите снова.")
        return

    print(f"Найдено файлов: {len(images)}\n")

    for i, image_path in enumerate(images, 1):
        print(f"[{i}/{len(images)}] {image_path.name}")
        process_document(image_path)

    print(f"\nГотово. Результаты в папке '{OUTPUT_DIR}'.")


if __name__ == "__main__":
    main()