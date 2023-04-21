class BrokenImageException(Exception):
    def __init__(self, image):
        self.image = image
        self.message = f"Unable to read image data"
        super().__init__(self.message)
