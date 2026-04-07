import os
from typing import Type

import cupy as cp
import customtkinter as ctk

from neural import NetworkDense

# from gui.frames.menu import MenuFrame
# from gui.frames.trainer import TrainerFrame
from app_ui.frames.scanner import ScannerFrame

class App(ctk.CTk):
    
    WIN_TITLE = "Handwritten Digit Recogniser Thingy"
    RESOLUTION = '1920x1080'
    PRIMARY_BTN_COLOUR = '#1F6AA5'
    LBL_CORNER_RADIUS = 10

    UPDATE_INTERVAL = 1000 # ms
    MODEL_DIR = r'C:\Users\ikram\Documents\Schoolwork\compsci\NEA 2026\src\training\models'
    
    DIGIT_IMG_SIZE = (28, 28)

    container: ctk.CTkFrame
    loaded_frames: dict[type[ctk.CTkFrame], ctk.CTkFrame]
    network: NetworkDense
    net_info: dict


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


    def show_frame(self, frame_class: type[ctk.CTkFrame]):
        '''Bring the specified frame to the front of container, displaying it.'''
        
        frame = self.loaded_frames[frame_class]
        frame.tkraise() # Bring to front/display


    def set_network(self, net_name: str) -> dict:
        '''
        Set the active network to the one described by the specified network name,
        and return the metadata about that network and its training.
        '''
        
        # To keep linter happy
        net_fname = ''

        # Get the file name of network from just the name itself
        # Raise error if file not found
        pretrained_nets_info = self.get_pretrained_networks()
        for i, pre_net_name in enumerate(pretrained_nets_info['net_names']):
            if pre_net_name == net_name:
                net_fname = pretrained_nets_info['net_fnames'][i]
            else:
                raise FileNotFoundError()

        # Load network container
        net_cont = cp.load(file=rf'{App.MODEL_DIR}\{net_fname}')
        assert type(net_cont) == dict # To keep linter happy

        # Change active network
        self.network = NetworkDense.load_pretrained(net_cont)
        
        # Save active network info as controller attributes so all frames can access it
        # Mirrors structure in train/train_networks.py
        self.net_info = {
            'f_name': net_fname,
            'net_name': net_cont['name'],
            'net_acc': net_cont['avg_acc'],
            'avg_loss': net_cont['avg_loss'],
            'layers': net_cont['layers']
        }
        
        # Return metadata
        return net_cont['metadata']
    

    def get_pretrained_networks(self) -> dict[str, list[str]]:
        '''Get dict of extensionless file names, net names, net accuracies, timestamps of trained networks.'''
        
        net_info = {
            # Get file names at cls-specified directory
            'f_names': os.listdir(path=App.MODEL_DIR),
            'net_names' : [],
            'net_accs': [],
            'timestamps': [],
        }

        for f_name in net_info['f_names']:
            print(f'FILE NAME: {f_name}')

            # Separator has to mirror what is in train/train_networks.py
            # Underscores are fine because ISO8601 uses T as separator instead of _
            f_name_split = f_name.split('_')

            net_info['net_names'].append(f_name_split[0])
            net_info['net_accs'].append(f_name_split[1])
            net_info['timestamps'].append(f_name_split[2])

        return net_info
    

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

        for F in (ScannerFrame,):
            frame = F(container=self.container, controller=self)
            self.loaded_frames[F] = frame
            frame.grid(row=0, column=0, sticky='nsew')

        for key, val in self.loaded_frames.items():
            print(key, val)
            print(type(key), type(val))

    
    