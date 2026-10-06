import shutil
from pathlib import Path

# --- НАСТРОЙКИ ---
# Замените этот путь на тот, где лежит распакованный датасет
SOURCE_DIR = Path(r"C:\Users\Мария\Downloads\archive\CAN\HK888152")

# Куда складывать картинки (папка внутри вашего проекта)
DEST_DIR = Path("dataset/images")
DEST_DIR.mkdir(parents=True, exist_ok=True)

# Счётчик
count = 0

# Рекурсивно обходим все файлы во всех вложенных папках
for img_path in SOURCE_DIR.rglob("*.jpg"):
    # Формируем новое имя: img_001.jpg, img_002.jpg и т.д.
    count += 1
    new_name = f"img_{count:03d}.jpg"  # :03d означает "три цифры с ведущими нулями"
    dest_path = DEST_DIR / new_name

    # Копируем файл
    shutil.copy2(img_path, dest_path)

    # Печатаем, что скопировали (чтобы видеть прогресс)
    print(f"{img_path.name}  ->  {new_name}")

print(f"\nГотово! Скопировано {count} картинок в {DEST_DIR}")