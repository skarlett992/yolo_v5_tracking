# yolo_v5_tracking
## требования к версии

- OS Ubuntu 20
- Python 3.9.6
- Pip 21.1.2

### Для запуска трекинга видео, его необходимо добавить в проект, 
### Видео доступно по ссылке 'https://disk.yandex.ru/i/4M8oK91WSkjA3Q'

# трекинг персон с помощью deep-sort:

Этот репозиторий содержит двухэтапный трекер: 
1. Детекция людей осуществляется с помощью семейства архитектур сетей YOLOv5.
YOLOv5 🚀 — это семейство архитектур и моделей обнаружения объектов, предварительно обученных на наборе данных COCO, 
и представляет собой исследование Ultralytics ('https://pytorch.org/hub/ultralytics_yolov5/')
с открытым исходным кодом ('https://github.com/ultralytics/yolov5.git')

2. Результаты детекции передаются алгоритму глубокой сортировки, который осуществляет трекинг, отслеживая объекты. 
    Запуск алгоритма происходит с помощью src/detectors.py - обращается к src/deep_sort.py осуществления трекинга
   ('https://kaiyangzhou.github.io/deep-person-reid/user_guide.html'),
 ссылка на репозиторий: ('https://github.com/KaiyangZhou/deep-person-reid')


# Параметры запуска (указать в конфигураторе):
Обязательные параметры:
--source Peoples.mp4 
{указать путь к видеопотоку для детекции}
--yolo_model weights/crowdhuman_yolov5m.pt 
{параметр конфигуратора - вес модели, нацеленной на обнаружение людей}
Дополнительные параметры:
--show-vid 
{при необходимости просматривать результаты трекинга в режиме онлайн}
--save-vid 
{при необходимости сохранить результаты трекинга}
--project inference/output
{директория для сохраненных результатов трекинга}

## Результат работы видеопотока сохраняется в папке в директории 'inference/output'

## Установка проекта: 
Установить VirtualBox на PC
Скачать по ссылке, импортировать его в VirtualBox и запустить образ. 

Реквизиты sudoer-a:
```
natalia : 123
```

Открыть терминал и перейти в директорию с проектом:
```commandline
cd ~/yolo_v5_tracking
```
Активировать окружение проекта:
```commandline
source .venv/bin/activate
```

Скачать через веб-браузер видео-файл и разместить его в корневой папке проекта:
https://drive.google.com/file/d/1I0Wcmys1uJZL3pkPlexT1zHDniMedluW/view?usp=sharing

Скачать через веб-браузер файл и установить его в директорию проекта `weights`:
https://drive.google.com/file/d/1nyv8oX2Hu2Huylc7yek6qtaCxNXkJeks/view?usp=sharing

## Запуск проекта
а)
Запустить через терминал команду:
```commandline
python main.py --source Peoples.mp4 \
                --yolo_model weights/crowdhuman_yolov5m.pt \
                --show-vid \
                --save-vid \
                --project inference/output
```
б)
На рабочем столе запустить файл `run.sh`

