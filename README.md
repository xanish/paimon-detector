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

## Improvements

- Preferably more training data
- Maybe find an alternative to lbpcascade_animeface, since on testing it doesn't seem to be as good in detecting faces
- Detect more characters?
