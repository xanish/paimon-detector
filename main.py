import json
import numpy as np
import os
import tensorflow as tf
from exceptions.face_not_detected import FaceNotDetected
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
    faces = []

    try:
        waifu = Waifu(work_dir='dataset')

        # extract faces from image
        faces = (waifu
                 .detect_faces(file)
                 .resize_faces(RESIZE_DIMENSIONS)
                 .get_faces())
    except FaceNotDetected as fnd:
        # this is super messy (find a way to clean it up)
        # just need a fallback to force predict even if
        # faces are not detected
        try:
            faces = [helpers.resize_image(fnd.image)]
        except Exception as error:
            raise HTTPException(status_code=500, detail=str(error))
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))

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

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    run(app, host='0.0.0.0', port=port)
