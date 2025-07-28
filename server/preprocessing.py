import cv2 as cv

def scale_image(self, image):
    image = cv.resize(image, None, fx=self.scale, fy=self.scale, interpolation=cv.INTER_LINEAR)
    return image