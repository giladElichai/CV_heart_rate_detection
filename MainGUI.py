# app.py - Main application setup

import tkinter as tk
from Controller import Controller
# from model import Model
from heartRateExtractor import CHeartRateExtractor
class App:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Webcam App")

        # Initialize MVC components
        self.model = CHeartRateExtractor()
        self.controller = Controller(self.root, self.model)

        # Start the main loop
        self.root.mainloop()

        # Clean up
        self.controller.release_webcam()

if __name__ == "__main__":
    app = App()
