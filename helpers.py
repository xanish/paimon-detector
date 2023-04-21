import cv2
import imageio
import os
import ulid
import urllib
import numpy as np

from io import BytesIO
from PIL import Image


def download_image(url: str, save_dir: str) -> str:
    request = urllib.request.Request(
        url=url,
        headers={"User-Agent": "Mozilla/5.0"}
    )

    file = os.path.join(save_dir, f'base-{ulid.new().int}')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    img_data = urllib.request.urlopen(request).read()
    image = Image.open(BytesIO(img_data))

    # convert gif to jpeg
    if image.format == 'GIF':
        open(f'{file}.gif', 'wb+').write(img_data)
        frames = imageio.mimread(f'{file}.gif')
        cv2.imwrite(f'{file}.jpg', cv2.cvtColor(frames[0], cv2.COLOR_RGB2BGR))
    else:
        img_array = np.asarray(bytearray(img_data), dtype=np.uint8)
        image = cv2.imdecode(img_array, -1)
        cv2.imwrite(f'{file}.jpg', image)

    return f'{file}.jpg'
