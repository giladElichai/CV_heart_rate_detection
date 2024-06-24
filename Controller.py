from functools import partial

import cv2
import PIL.Image
import PIL.ImageTk
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


class Controller:
    def __init__(self, root, model):
        self.root = root
        self.model = model
        
        self.work = False 
        # Initialize the webcam
        self.cap = cv2.VideoCapture(0)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.model.set_fps(self.fps)
        
        self.iFrame = 0
        
        # Initialize the GUI components
        self.setup_gui()

    def setup_gui(self):
        # Create a label for webcam feed
        self.webcam_label = tk.Label(self.root)
        self.webcam_label.grid(row=0, column=0, padx=10, pady=10)

        # Create Matplotlib figure and canvas for line graph
        self.fig = Figure(figsize=(7, 2), dpi=100)
        self.plot = self.fig.add_subplot(111)
        self.plot.set_xlabel('X Label')
        self.plot.set_ylabel('Y Label')
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().grid(row=1, column=0, padx=10, pady=10)

        # Create checkboxes
        self.checkbox_frame = tk.Frame(self.root)
        self.checkbox_frame.grid(row=0, column=1, rowspan=2, padx=10, pady=10)
        
        self.checkbox1_var = tk.IntVar()
        self.checkbox2_var = tk.IntVar()
        self.checkbox3_var = tk.IntVar()

        self.checkbox1 = tk.Checkbutton(self.checkbox_frame, text="Start", variable=self.checkbox1_var,
                                        command=partial(self.checkbox_selection, 1))
        self.checkbox1.grid(row=0, column=1, padx=10, pady=10, sticky='w')
        
        # self.checkbox2 = tk.Checkbutton(self.checkbox_frame, text="Checkbox 2", variable=self.checkbox2_var,
        #                                 command=partial(self.checkbox_selection, 2))
        # self.checkbox2.grid(row=1, column=1, padx=10, pady=10, sticky='w')

        # self.checkbox3 = tk.Checkbutton(self.checkbox_frame, text="Checkbox 3", variable=self.checkbox3_var,
        #                                 command=partial(self.checkbox_selection, 3))
        # self.checkbox3.grid(row=2, column=1, padx=10, pady=10, sticky='w')
        
        # # Create a label and entry widget for program output
        self.output_label = tk.Label(self.checkbox_frame, text="HR:")
        self.output_label.grid(row=1, column=0, padx=10, pady=10, sticky='w')

        self.output_entry_var = tk.StringVar()
        self.output_entry = tk.Entry(self.checkbox_frame, textvariable=self.output_entry_var, state='readonly')
        self.output_entry.grid(row=1, column=1, padx=10, pady=10, sticky='we')
        self.output_entry_var.set(str(0))


        # Start webcam feed and line graph updates
        self.update_webcam_feed()
        self.update_line_graph()

    def update_webcam_feed(self):
        ret, frame = self.cap.read()
        if ret:
            if self.work:
                frame, bpm = self.model.process(self.iFrame, frame) 
                self.output_entry_var.set(str(bpm)) 
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
            self.webcam_label.imgtk = photo
            self.webcam_label.configure(image=photo)
            self.iFrame += 1   
            self.update_line_graph()
        
        self.webcam_label.after(10, self.update_webcam_feed)

    def update_line_graph(self):
        # Placeholder for updating line graph
        # self.plot.plot(fft)
        if not self.work:
            return 
        freq, fft = self.model.plot_bpm()
        if len(fft) < 30:
            return 
        self.plot.clear()  # Clear previous plot
        self.plot.plot(fft[-100:])  # Plot new data
        self.plot.set_xlabel('time')
        self.plot.set_ylabel('amp')
        self.canvas.draw() 
        pass

    def checkbox_selection(self, checkbox_id):
        # Placeholder for handling checkbox selection
        self.work = bool(self.checkbox1_var.get())
        
        pass

    def release_webcam(self):
        self.cap.release()
        cv2.destroyAllWindows()