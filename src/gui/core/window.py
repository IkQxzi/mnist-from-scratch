import os

import numpy as np
import cupy as cp

# from neural import NetworkDense
import customtkinter as ctk
from customtkinter import CTkImage, CTkLabel, CTkButton, CTkComboBox
from customtkinter import filedialog
from PIL import Image, ImageTk, ImageOps, ImageFile

from neural.core.network import NetworkDense

class App(ctk.CTk):

    WIN_TITLE = "Handwritten Digit Recogniser Thingy"
    RESOLUTION = '1920x1080'
    IMG_SIZE = (500, 500)
    DIGIT_IMG_SIZE = (28, 28)
    SKIP_IMG_SELECT = False
    PRETRAINED_NET_PATH = r'C:\Users\ikram\Documents\Schoolwork\compsci\NEA 2026\src\training\models'
    UPDATE_INTERVAL = 1000 # in ms

    def __init__(self): # fix net later
        super().__init__() # keep App happy

        # net stuff
        self.network: NetworkDense
        self.curr_digit: Image.Image
        self.curr_digit_arr: np.ndarray # 28x28

        # settings
        self.title(App.WIN_TITLE)
        self.geometry(App.RESOLUTION)
        self.is_img_displayed = False


        # widgets
        self.display_widgets()
        
        self.curr_network = self.net_selection.get()
        print(self.curr_network)

 
    def display_widgets(self):

        # Buttons
        self.import_btn = CTkButton(self, text="Import Image", command=self.display_img)
        self.import_btn.grid(row=0, column=0, padx=20, pady=20)
        self.import_btn.grid(sticky="n")

        self.scan_btn = CTkButton(self, text="Scan Image", command=self.scan_img)
        self.scan_btn.grid(row=2, column=0, padx=20, pady=20)
        self.scan_btn.grid(sticky='n')

        # will go row 1, col 0
        self.img_label = CTkLabel(self, text="")

        # Selection of pretrained networks to choose from
        # Fill with values first to not have 'CTkComboBox' as the default value
        self.net_selection = CTkComboBox(self, values=self.get_pretrained_networks())
        self.update_net_selection() # constantly recalls itself
        self.net_selection.grid(row=2, column=2, padx=20, pady=20)

        # network outputs (set to grey)
        self.net_pred_lbl = CTkLabel(self, text="The network's prediction will be displayed here.")
        self.net_pred_lbl.grid(row=3, column=0, padx=20, pady=5)
        
        self.net_conf_lbl = CTkLabel(self, text="The network's confidence will be displayed here.")
        self.net_conf_lbl.grid(row=4, column=0, padx=20, pady=5)

        # do last
        self.gpu_usage_lbl = CTkLabel

    
    def update_net_selection(self) -> None:
        '''Update the display of available networks every time interval.'''

        self.net_selection.configure(values=self.get_pretrained_networks())
        self.after(App.UPDATE_INTERVAL, self.update_net_selection)


    def get_pretrained_networks(self) -> list[str]:
        '''Get list of file names (without extension) of pretrained networks available for use.'''
        
        file_names = os.listdir(path=App.PRETRAINED_NET_PATH)

        clean_f_names = [file_name.rstrip('.npz') for file_name in file_names]

        return clean_f_names


    def update_net_lbls(self, new_pred: int, new_conf: float):
        '''Update the displayed network prediction and confidence score (as percentage).'''

        self.net_pred_lbl.configure(text=f'The network predicts this number is a: {new_pred}')
        self.net_conf_lbl.configure(text=f"The network is {new_conf}% confident in its prediction.")


    # Don't yet know what this will return
    def get_net_outputs(self, inputs: cp.ndarray):
        net_probs = self.network.fwd(inputs)


    def display_img(self):
        '''Display img selected by user.'''
        
        img = self.import_img()

        if img:
            self.img_label.configure(image=img)

            self.img_label.grid(row=1, column=0, padx=20, pady=20)
            self.img_label.grid(sticky="n")

            self.is_img_displayed = True

        # Will want to compress to 28x28 later
        # print(f'Image size: {img.size}')

        return None


    def import_img(self) -> CTkImage | None:
        '''Prompt user to import img and return it if selected.'''

        if App.SKIP_IMG_SELECT:
            img_path = r"C:\Users\ikram\Documents\Schoolwork\compsci\test_img.png"

        else:
            img_path = filedialog.askopenfilename(title="Open Image File", 
                                          filetypes=[("Image files", "*.png *.jpg *.jpeg *.gif *.bmp *.ico")])


        if img_path:

            with Image.open(img_path) as img_obj:

                # Load the img into RAM instead of keeping file open
                img_obj.load()

                # Save for use later with NN
                self.curr_digit = img_obj

                img = CTkImage(img_obj, size=App.IMG_SIZE)

            return img
        
        return None


    def conv_img_np(self, img: Image.Image) -> np.ndarray:
        '''Convert an image to a normalised np arr [0, 1].'''

        img = img.resize(App.DIGIT_IMG_SIZE)
        img_grey = img.convert(mode='F')

        img_arr = np.asarray(img_grey)
        img_arr = img_arr / 255
        print(img_arr)

        return img_arr

    
    def scan_img(self):
        '''Pass image into neural network and display output.'''

        self.curr_digit_arr = self.conv_img_np(self.curr_digit)

        # get img from self
        # pass through nn
        # display output


        pass





    

# Run the app
app = App()
app.mainloop()

