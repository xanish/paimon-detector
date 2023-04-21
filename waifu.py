import os
import random
import time
import cv2


class Waifu:
    def __init__(self, work_dir, cascade_file='lbpcascade_animeface.xml', debug=False):
        if not os.path.isfile(cascade_file):
            raise RuntimeError(f"{cascade_file}: not found")

        self.work_dir = work_dir
        self.debug = debug
        self.cascade = cv2.CascadeClassifier(cascade_file)
        self.faces = []
        self.resized_faces = []

    def detect_faces(self, file):
        self.faces = []
        image = cv2.imread(file, cv2.IMREAD_COLOR)

        if image is not None:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            faces = self.cascade.detectMultiScale(
                gray,
                scaleFactor=1.01,
                minNeighbors=3,
                minSize=(20, 20)
            )

            if len(faces) == 0:
                raise ValueError(f"No face detected in {file}")

            for (x, y, w, h) in faces:
                self.faces.append(image[y:y+h, x:x+w])
                if self.debug:
                    # highlight the detected face with a rectangle
                    cv2.rectangle(
                        image,
                        (x, y),
                        (x + w, y + h),
                        (0, 0, 255),
                        2
                    )

            if self.debug:
                # write the image as output
                debug_file = os.path.join(
                    self.work_dir,
                    f'debug/debug-{time.time()}-{random.randint(0, 1000)}.jpg'
                )
                cv2.imwrite(debug_file, image)
        else:
            raise ValueError(f"Broken image {file}")

        return self

    def resize_faces(self, dimensions):
        self.resized_faces = []
        for face in self.faces:
            image = cv2.resize(
                face,
                dimensions,
                interpolation=cv2.INTER_AREA
            )
            self.resized_faces.append(image)

        return self

    def save_faces(self, name, dir):
        faces = self.resized_faces if len(
            self.resized_faces) > 0 else self.faces

        base_path = os.path.join(self.work_dir, dir)
        if not os.path.exists(base_path):
            os.makedirs(base_path)

        for i, face in enumerate(faces):
            cv2.imwrite(os.path.join(base_path, f'{name}.jpg'), face)

    def get_faces(self):
        return self.resized_faces if len(self.resized_faces) > 0 else self.faces
