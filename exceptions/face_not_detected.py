class FaceNotDetected(Exception):
    def __init__(self, image):
        self.image = image
        self.message = f"Unable to detect face in image"
        super().__init__(self.message)
