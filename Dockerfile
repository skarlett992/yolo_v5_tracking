FROM python:3.9.6-slim as builder

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    apt-get clean

RUN python -m pip install --upgrade pip

WORKDIR /app
COPY requirements_dev.txt ./requirements.txt
RUN pip wheel --no-cache-dir \
              --no-deps \
              --wheel-dir /app/wheels  \
              --default-timeout=100 \
              -r requirements.txt

FROM python:3.9.6-slim as yolo_v5_test_project
ENV QT_DEBUG_PLUGINS=1
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg libsm6 libxext6 libqt5gui5 && \
    apt-get clean

WORKDIR /app

COPY --from=builder /app/wheels /wheels
COPY --from=builder /app/requirements.txt .

RUN pip install --no-cache /app/wheels/*