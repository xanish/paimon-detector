import os

from waifu import Waifu


SRC_DIR = 'dataset/raw'
RESIZE_DIMENSIONS = (96, 96)


def generate_processed_dataset():
    wfu = Waifu(work_dir='dataset')
    waifus = os.listdir(SRC_DIR)

    for waifu in sorted(waifus):
        waifu_images = os.listdir(os.path.join(SRC_DIR, waifu))

        for i, waifu_img in enumerate(sorted(waifu_images)):
            image = os.path.join(SRC_DIR, waifu, waifu_img)
            try:
                (wfu.detect_faces(image)
                    .resize_faces(RESIZE_DIMENSIONS)
                    .save_faces(f'{waifu}-{i}', f'processed/{waifu}'))
            except Exception as error:
                print(f"Caught: {error}")


if __name__ == '__main__':
    generate_processed_dataset()
