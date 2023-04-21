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
{
  "faces_detected":true,
  "results":[
    { "possible": "not_paimon", "chances": 99.98887777328491 }
  ]
}
```

## Datasets

- [Genshin impact heads](https://www.kaggle.com/datasets/honchokomodo/genshin-impact-heads)
- [Anime character face dataset](https://www.kaggle.com/datasets/thedevastator/anime-face-dataset-by-character-name)
- Some handpicked images from google

## Training results

- Epoch 01: loss: 0.7328 - accuracy: 0.5698 - val_loss: 0.7520 - val_accuracy: 0.4884
- Epoch 02: loss: 0.6573 - accuracy: 0.5756 - val_loss: 0.6219 - val_accuracy: 0.8605
- Epoch 03: loss: 0.5537 - accuracy: 0.8488 - val_loss: 0.4925 - val_accuracy: 0.7674
- Epoch 04: loss: 0.3847 - accuracy: 0.9012 - val_loss: 0.3806 - val_accuracy: 0.8140
- Epoch 05: loss: 0.2630 - accuracy: 0.9070 - val_loss: 0.2487 - val_accuracy: 0.8605
- Epoch 06: loss: 0.2311 - accuracy: 0.9244 - val_loss: 0.1846 - val_accuracy: 0.9302
- Epoch 07: loss: 0.1646 - accuracy: 0.9593 - val_loss: 0.1628 - val_accuracy: 0.9302
- Epoch 08: loss: 0.1503 - accuracy: 0.9477 - val_loss: 0.1206 - val_accuracy: 0.9767
- Epoch 09: loss: 0.1110 - accuracy: 0.9709 - val_loss: 0.1470 - val_accuracy: 0.9302
- Epoch 10: loss: 0.0973 - accuracy: 0.9709 - val_loss: 0.1195 - val_accuracy: 0.9767

## Improvements

- Preferably more training data
- Detect more characters?
