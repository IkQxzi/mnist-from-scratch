
# internal
from typing import TYPE_CHECKING

# 3rd party
import cupy as cp
import customtkinter as ctk
from PIL import Image

# local

if TYPE_CHECKING:
    from gui.controller import App

class ScannerFrame(ctk.CTkFrame):

    USR_IMG_SIZE = (500, 500)
    SKIP_IMG_SELECT = False
    controller: App

    _curr_img_pil: Image.Image
    _curr_img_ctk: ctk.CTkImage
    _curr_img_arr: cp.ndarray # 28x28
    _is_img_displayed: bool

    # Widgets
    _import_btn: ctk.CTkButton
    _scan_btn: ctk.CTkButton
    _usr_img_label: ctk.CTkLabel
    _net_selector: ctk.CTkComboBox
    _net_pred_lbl: ctk.CTkLabel
    _net_conf_lbl: ctk.CTkLabel
    _gpu_usage_lbl: ctk.CTkLabel

    def __init__(self, container, controller: App):
        
        # Put frame inside of container
        super().__init__(master=container)

        self.controller = controller

        self._create_widgets()
        self._layout_widgets()

    
    def _create_widgets(self):
        # Usr buttons
        self._import_btn = ctk.CTkButton(self, text="Import Image", command=self._on_import_click)
        self._scan_btn = ctk.CTkButton(self, text="Scan Image", command=self._on_scan_click)

        # img
        self._usr_img_lbl = ctk.CTkLabel(self, text="")

        # Fill with values first to not have 'CTkComboBox' as the default value
        self._net_selector = ctk.CTkComboBox(self, values=self.controller.get_pretrained_networks())

    
    # Terrible layout, fix later
    def _layout_widgets(self):
    
        # btns
        self._import_btn.grid(row=0, column=0, padx=20, pady=20, sticky='n')
        self._scan_btn.grid(row=2, column=0, padx=20, pady=20, sticky='n')

        # img
        self._usr_img_lbl.grid(row=1, column=0, padx=20, pady=20)

        # net outputs
        self._net_selector.grid(row=2, column=2, padx=20, pady=20)
        self._net_pred_lbl.grid(row=3, column=0, padx=20, pady=5)
        self._net_conf_lbl.grid(row=4, column=0, padx=20, pady=5)

        # TODO: add gpu lbl here


    def reset_frame(self):
        # Remove the image and stuff
        pass




    def _conv_img_to_arr(self) -> cp.ndarray:
            
        img_grey_resized = self._curr_img_pil.resize(self.controller.DIGIT_IMG_SIZE).convert(mode='F')
        img_arr = cp.asarray(img_grey_resized) / 255

        return img_arr


    def _on_import_click(self):

        # temp for easier testing
        if ScannerFrame.SKIP_IMG_SELECT:
            img_path = r"C:\Users\ikram\Documents\Schoolwork\compsci\test_img.png"
        else:
            img_path = ctk.filedialog.askopenfilename(title="Open Image File", 
                                          filetypes=[("Image files", "*.png *.jpg *.jpeg *.gif *.bmp *.ico")])

        if not img_path: return

        self._curr_img_pil =  Image.open(img_path) 
        self._curr_img_ctk = ctk.CTkImage(self._curr_img_pil, size=ScannerFrame.USR_IMG_SIZE)

        self._usr_img_label.configure(image=self._curr_img_ctk)


    def _on_scan_click(self):
        
        if not self._curr_img_pil:
            return

        img_arr = self._conv_img_to_arr()

        # TODO: network outputs
        self.controller.network.fwd(img_arr)