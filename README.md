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

- Epoch 1: loss: 0.6575 - accuracy: 0.5622 - val_loss: 0.6494 - val_accuracy: 0.5800
- Epoch 2: loss: 0.5668 - accuracy: 0.7164 - val_loss: 0.5070 - val_accuracy: 0.7400
- Epoch 3: loss: 0.3766 - accuracy: 0.8557 - val_loss: 0.3863 - val_accuracy: 0.7800
- Epoch 4: loss: 0.3228 - accuracy: 0.8706 - val_loss: 0.4574 - val_accuracy: 0.7200
- Epoch 5: loss: 0.3144 - accuracy: 0.8756 - val_loss: 0.2726 - val_accuracy: 0.8600

## Improvements

- Preferably more training data
- Detect more characters?
