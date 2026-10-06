# DocAI

Утилита для выделения фотографии и машиночитаемой зоны (MRZ) на изображениях документов. Детектор YOLO находит нужные области, сохраняет их отдельно и распознаёт текст MRZ с помощью Tesseract OCR.

## Возможности

- обрабатывает изображения JPG, JPEG, PNG и BMP;
- выделяет зоны `photo` и `mrz` обученной моделью YOLO;
- сохраняет исходный документ и вырезанные области;
- распознаёт MRZ на английском языке;
- обрабатывает сразу все изображения из указанной папки;
- работает на macOS, включая Apple Silicon.

## Требования

- macOS;
- Python 3.11 или новее;
- Homebrew;
- Tesseract OCR с английскими языковыми данными.

## Установка

В корневой папке проекта выполните:

```bash
brew install tesseract
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Проверьте, что Tesseract доступен:

```bash
tesseract --version
```

На компьютерах с Apple Silicon исполняемый файл обычно находится в `/opt/homebrew/bin/tesseract`. На Intel Mac типичный путь — `/usr/local/bin/tesseract`.

## Обработка документов

1. Создайте папку `incoming` в корне проекта, если она ещё не создана.
2. Поместите в неё изображения документов.
3. Запустите:

```bash
python main.py
```

По умолчанию скрипт использует модель `model/best.pt` и сохраняет результаты в папку `output`.

### Свои папки и модель

Пути можно передать явно:

```bash
python main.py \
  --input ~/Documents/passports \
  --output ~/Documents/passport-results \
  --model model/best.pt
```

Если Tesseract не добавлен в `PATH`, укажите путь к нему:

```bash
python main.py --tesseract /opt/homebrew/bin/tesseract
```

## Результаты

Для каждого исходного файла создаётся отдельная папка в `output`:

```text
output/
└── passport_001/
    ├── passport_001.jpg  # исходное изображение
    ├── face.jpg          # вырезанная фотография
    ├── mrz.jpg           # вырезанная MRZ
    ├── mrz.txt           # результат OCR
    └── boxes.txt         # класс, уверенность и координаты областей
```

Если модель не найдёт одну из областей, соответствующий файл не будет создан. Координаты всех найденных областей сохраняются в `boxes.txt`.

## Подготовка изображений для датасета

Команда рекурсивно найдёт JPEG-файлы в указанной папке и скопирует их в `dataset/images`, переименовав в `img_001.jpg`, `img_002.jpg` и так далее:

```bash
python collect_images.py /путь/к/распакованному/датасету
```

Другую папку назначения можно передать параметром `--destination`:

```bash
python collect_images.py /путь/к/датасету --destination dataset/images
```

## Обучение модели

Для обучения используются веса `yolo26n.pt` и конфигурация `data.yaml`:

```bash
python train.py
```

На Apple Silicon обучение использует Metal Performance Shaders (`mps`). На Intel Mac или при проблемах с MPS используйте CPU:

```bash
python train.py --device cpu
```

Полный список параметров любой команды доступен через `--help`:

```bash
python main.py --help
python train.py --help
python collect_images.py --help
```

## Структура проекта

```text
.
├── main.py               # обработка документов
├── train.py              # обучение YOLO-модели
├── collect_images.py     # подготовка изображений датасета
├── data.yaml             # конфигурация классов и датасета
├── model/best.pt         # обученные веса детектора
├── examples/             # примеры результатов
└── requirements.txt      # Python-зависимости
```
