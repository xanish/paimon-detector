import json
import numpy as np
import os
import tensorflow as tf
import helpers

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from uvicorn import run
from waifu import Waifu


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
    file = helpers.download_image(url, TEMP_DIR)

    try:
        waifu = Waifu(work_dir='dataset')
        # extract faces from image
        faces = (waifu
                 .detect_faces(file)
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
