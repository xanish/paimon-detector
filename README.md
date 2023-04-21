# Paimon Detector

Simple paimon detector using python, opencv ([lbpcascade_animeface](https://github.com/nagadomi/lbpcascade_animeface) for detecting faces) and tensorflow.

Project packaged using simple rest api which takes a `url` to any image and tries to detect whether it has paimon or not.

- Why do you need a paimon detector?
  - Well this is to prevent the recent influx and spam of paimon gif/images on discord.
- Why would someone spam paimon images?
  - I'm not sure ask them.

## Setup Instructions

- Run `pip install -r requirements.txt`
- Run `python3 -m uvicorn main:app --reload` or `python3 main.py`

Alternatively you can run the app using docker

- Run `docker build -t paimon .`
- Run `docker run -d -p 8000:8000 paimon`

## Example Request

```curl
curl --location --request GET 'http://localhost:8000/has-paimon?url=https://example.com/some/image.jpg'
```

## Response Format

Will contain an item for each detected face in the image

```json
[
  { "possible":"paimon", "chances": 93.1159257888794 }
]
```

## Datasets

- [Genshin impact heads](https://www.kaggle.com/datasets/honchokomodo/genshin-impact-heads)
- [Anime character face dataset](https://www.kaggle.com/datasets/thedevastator/anime-face-dataset-by-character-name)
- Some handpicked images from google

## Training results

- Epoch 01: loss: 0.7451 - accuracy: 0.5224 - val_loss: 0.6945 - val_accuracy: 0.4600
- Epoch 02: loss: 0.6538 - accuracy: 0.5522 - val_loss: 0.6153 - val_accuracy: 0.5800
- Epoch 03: loss: 0.5546 - accuracy: 0.7015 - val_loss: 0.5308 - val_accuracy: 0.8600
- Epoch 04: loss: 0.4264 - accuracy: 0.8557 - val_loss: 0.4412 - val_accuracy: 0.7600
- Epoch 05: loss: 0.2875 - accuracy: 0.9055 - val_loss: 0.2435 - val_accuracy: 0.9000
- Epoch 06: loss: 0.1818 - accuracy: 0.9353 - val_loss: 0.3181 - val_accuracy: 0.8800
- Epoch 07: loss: 0.2242 - accuracy: 0.9204 - val_loss: 0.4449 - val_accuracy: 0.8400
- Epoch 08: loss: 0.2087 - accuracy: 0.9254 - val_loss: 0.1941 - val_accuracy: 0.9600
- Epoch 09: loss: 0.1031 - accuracy: 0.9751 - val_loss: 0.2222 - val_accuracy: 0.9200
- Epoch 10: loss: 0.0915 - accuracy: 0.9751 - val_loss: 0.1607 - val_accuracy: 0.9200

## Improvements

- Preferably more training data
- Detect more characters?
