import os
from typing import Type

import customtkinter as ctk

from neural import NetworkDense

# from gui.frames.menu import MenuFrame
# from gui.frames.trainer import TrainerFrame
from gui.frames.scanner import ScannerFrame

class App(ctk.CTk):
    
    WIN_TITLE = "Handwritten Digit Recogniser Thingy"
    RESOLUTION = '1920x1080'
    UPDATE_INTERVAL = 1000 # ms
    MODEL_DIR = r'C:\Users\ikram\Documents\Schoolwork\compsci\NEA 2026\src\training\models'
    
    DIGIT_IMG_SIZE = (28, 28)

    container: ctk.CTkFrame
    loaded_frames: dict[type[ctk.CTkFrame], ctk.CTkFrame]
    network: NetworkDense

    def __init__(self):
        '''
        Docstring for __init__
        
        :param frame_list: List of 
        :type frame_list: list[ctk.CTkFrame]
        '''
        
        super().__init__()

        # TODO: load a default network
        # self.network = None

        self._setup_window()
        self._load_container()


    def _setup_window(self):
        self.title(App.WIN_TITLE)
        self.geometry(App.RESOLUTION)


    def _load_container(self):

        # Acts as a static parent for all app states
        # Prevents the need for other frames to be destroyed:
        # They can all be stacked on top of each other instead
        # And brought to foreground as required
        self.container = ctk.CTkFrame(self)
        
        self.container.pack(
            fill="both", # To make container fill both x & y directions in the window by default
            expand=True # To make container expand to fill the window if resized
        ) 

        # Make the grid inside the container (0, 0) expand to fill the container
        # By giving it 100% priority over free space (weight is distributed between grid items)
        self.container.grid_rowconfigure(index=0, weight=1)
        self.container.grid_columnconfigure(index=0, weight=1)

        self.loaded_frames = dict()

        # Initialise frames and put into self.loaded_frames

        frame = ScannerFrame(container=self.container, controller=self)
        self.loaded_frames[ScannerFrame] = frame

        for key, val in self.loaded_frames.items():
            print(key, val)
            print(type(key), type(val))

    
    def show_frame(self, frame_class: type[ctk.CTkFrame]):
        '''Bring the selected frame to the front of container displaying it.'''
        
        frame = self.loaded_frames[frame_class]
        frame.tkraise() # Bring to front/display

    
    def get_pretrained_networks(self) -> list[str]:
        '''Get list of file names (without extension) of pretrained networks available for use.'''
        
        file_names = os.listdir(path=App.MODEL_DIR)

        clean_f_names = [file_name.rstrip('.npz') for file_name in file_names]

        return clean_f_names
    
app = App()
app.mainloop()