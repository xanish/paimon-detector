FROM python:3.11-bullseye

EXPOSE 8000

RUN  mkdir -p  /paimon-detector

WORKDIR  /paimon-detector

# RUN apt-get update && \
#     apt-get install ffmpeg libsm6 libxext6 -y && \
#     pip install --no-cache-dir -U pip
RUN pip install --no-cache-dir -U pip

COPY requirements.txt .

RUN pip install -r requirements.txt

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

COPY  . .

CMD ["python3", "main.py"]
