import cv2
import json
import imageio
import numpy as np
import os
import random
import time
import tensorflow as tf
import urllib

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from uvicorn import run
from waifu import Waifu
from io import BytesIO
from PIL import Image

TEMP_DIR = 'dataset/temp'
RESIZE_DIMENSIONS = (96, 96)
CLASS_NAMES = json.load(open('models/paimon_classifier/class_labels.json'))

app = FastAPI()
model = tf.keras.models.load_model('models/paimon_classifier')

origins = ['*']
methods = ['*']
headers = ['*']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=methods,
    allow_headers=headers
)


@app.get('/has-paimon')
async def image_has_paimon(url: str = None):
    if url is None:
        return {"message": "url is required"}

    # download and save the image for processing
    request = urllib.request.Request(
        url=url,
        headers={"User-Agent": "Mozilla/5.0"}
    )

    img_data = urllib.request.urlopen(request).read()
    image = Image.open(BytesIO(img_data))

    file = os.path.join(
        TEMP_DIR, f'base-{time.time()}-{random.randint(0, 1000)}'
    )
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)

    # convert gif to jpeg
    if image.format == 'GIF':
        open(f'{file}.gif', 'wb+').write(img_data)
        frames = imageio.mimread(f'{file}.gif')
        cv2.imwrite(f'{file}.jpg', cv2.cvtColor(frames[0], cv2.COLOR_RGB2BGR))
    else:
        img_array = np.asarray(bytearray(img_data), dtype=np.uint8)
        image = cv2.imdecode(img_array, -1)
        cv2.imwrite(f'{file}.jpg', image)

    waifu = Waifu(work_dir='dataset')

    try:
        # extract faces from image
        faces = (waifu
                 .detect_faces(f'{file}.jpg')
                 .resize_faces(RESIZE_DIMENSIONS)
                 .get_faces())

        # run model on each detected face
        predictions = []
        for face in faces:
            face_array = tf.expand_dims(face, 0)
            result = model.predict(face_array)
            score = tf.nn.softmax(result)

            predictions.append({
                "possible": CLASS_NAMES[np.argmax(score)],
                "chances": 100 * np.max(score)
            })

        return predictions
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    run(app, host='0.0.0.0', port=port)
