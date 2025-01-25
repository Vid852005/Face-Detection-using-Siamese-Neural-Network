from keras.src.saving.legacy.save import load_model
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout

# Kivy UX components
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label

# Other Kivy stuff
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger

import cv2
import tensorflow as tf

# Custom module
from layer import L1Dist
import os
import numpy as np


class Camapp(App):
    def build(self):
        # UI components
        self.img1 = Image(size_hint=(1, 0.8))
        self.button = Button(text="Verify",on_press=self.verify, size_hint=(1, 0.1))
        self.verification_label = Label(text="Verification Uninitiated", size_hint=(1, 0.1))

        # Layout
        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.img1)
        layout.add_widget(self.button)
        layout.add_widget(self.verification_label)

        # Load model
        self.model = tf.keras.models.load_model('Siamese_model.keras', custom_objects={'L1Dist': L1Dist}, )

        # OpenCV video capture
        self.capture = cv2.VideoCapture(1)
        if not self.capture.isOpened():
            Logger.error("Camapp: Unable to access the camera")

        # Schedule update function
        Clock.schedule_interval(self.update, 1.0 / 33.0)

        # Connect button to verify method
        self.button.bind(on_press=self.verify)

        return layout

    def update(self, *args):
        ret, frame = self.capture.read()
        if ret:
            # Crop and process frame
            frame = frame[120:120 + 250, 200:200 + 250, :]

            # Convert image to texture
            buf = cv2.flip(frame, 0).tobytes()
            img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')

            # Display the texture
            self.img1.texture = img_texture

    def preprocess(self, file_path):
        byte_img = tf.io.read_file(file_path)
        img = tf.io.decode_jpeg(byte_img)
        img = tf.image.resize(img, (100, 100))
        img = img / 255.0
        return img

    def verify(self, *args):
        detection_threshold = 0.7
        verification_threshold = 0.8

        # Save input image
        SAVE_PATH = os.path.join('application_data', 'input_image', 'input_image.jpg')
        ret, frame = self.capture.read()
        if ret:
            frame = frame[120:120 + 250, 200:200 + 250, :]
            cv2.imwrite(SAVE_PATH, frame)

        # Perform verification
        results = []
        for image in os.listdir(os.path.join('application_data', 'verification_images')):
            input_img = self.preprocess(os.path.join('application_data', 'input_image', 'input_image.jpg'))
            validation_img = self.preprocess(os.path.join('application_data', 'verification_images', image))
            result = self.model.predict([np.expand_dims(input_img, axis=0), np.expand_dims(validation_img, axis=0)])

            results.append(result)

        # Calculate detection and verification
        detection = np.sum(np.array(results) > detection_threshold)
        verification = detection / len(os.listdir(os.path.join('application_data', 'verification_images')))
        verified = verification > verification_threshold

        # Update label
        self.verification_label.text = 'Verified' if verified else 'Unverified'
        return results, verified

    def on_stop(self):
        # Release OpenCV resources
        if self.capture.isOpened():
            self.capture.release()


if __name__ == '__main__':
    Camapp().run()
