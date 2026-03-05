
# internal
from typing import TYPE_CHECKING

# 3rd party
import cupy as cp
import customtkinter as ctk
from PIL import Image, ImageDraw, ImageGrab

# local
if TYPE_CHECKING:
    from app_ui.controller import App

class ScannerFrame(ctk.CTkFrame):

    USR_IMG_SIZE = (500, 500)
    SKIP_IMG_SELECT = False
    BTN_HEIGHT = 50
    BTN_FONT = ("Arial", 25, "bold")
    SMALL_BTN_HEIGHT = 20
    SMALL_BTN_FONT = ("Arial", 14, "bold")
    BRUSH_SIZE = 50
    controller: "App"

    _curr_img_pil: Image.Image
    _curr_img_ctk: ctk.CTkImage
    _curr_img_arr: cp.ndarray # 28x28
    _is_img_displayed: bool

    
    # Input widgets
    _inputs_tabs: ctk.CTkTabview
    
    # Network tab widgets
    _net_tab: ctk.CTkFrame
    _net_selector: ctk.CTkComboBox
    
    # Draw tab widgets
    _draw_tab: ctk.CTkFrame
    _draw_tab_txt_box: ctk.CTkLabel
    _canvas: ctk.CTkCanvas
    _pil_canvas_cpy: Image.Image
    _pil_canvas_cpy_draw: ImageDraw.ImageDraw
    _clear_btn: ctk.CTkButton
    _scan_btn_d_tab: ctk.CTkButton

    # Upload tab wijdgets
    _upload_tab: ctk.CTkFrame
    _upload_btn: ctk.CTkButton
    _scan_btn_u_tab: ctk.CTkButton
    _usr_img_lbl: ctk.CTkLabel
    
    # Results widgets
    _results_frame: ctk.CTkFrame
    _net_pred_lbl: ctk.CTkLabel
    _net_conf_lbl: ctk.CTkLabel
    _gpu_usage_lbl: ctk.CTkLabel
    

    def __init__(self, container, controller: "App"):
        
        # Put frame inside of container
        super().__init__(master=container)

        self.controller = controller

        # Split the frame into two equal columns & 1 row
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Input side split into tabs
        self._inputs_tabs = ctk.CTkTabview(self)
        self._create_net_tab()
        self._create_draw_tab()
        self._create_upload_tab()
        self._create_results_frame()

        self._inputs_tabs.grid(row=0, column=0, padx=20, pady=20, sticky='nsew')
        self._layout_net_tab()
        self._layout_draw_tab()
        self._layout_upload_tab()
        self._layout_results_frame()

    def reset_frame(self):
        # Remove the image and stuff
        pass


    def _create_net_tab(self):

        self._net_tab = self._inputs_tabs.add("Network")


    def _create_draw_tab(self):
        
        # Add to input tabs
        self._draw_tab = self._inputs_tabs.add("Draw")

        self._draw_tab_txt_box = ctk.CTkLabel(self._draw_tab, text="Try to fill the canvas as much as possible for the best network predictions.",
                                             height=self.SMALL_BTN_HEIGHT, font=self.SMALL_BTN_FONT, corner_radius=15, bg_color='#1F6AA5',) 

        # For user to draw on 
        self._canvas = ctk.CTkCanvas(self._draw_tab)

        # We must create a ghost copy of the canvas to avoid having to work with file exports
        # For now, use (1, 1) size; match to canvas in layout method
        self._pil_canvas_cpy: Image.Image
        self._pil_canvas_cpy_draw: ImageDraw.ImageDraw # Must be instantiated last
        
        # Moving the mouse should only draw when B1 (LMB) is held down
        self._canvas.bind('<B1-Motion>', self._canvas_draw)

        self._clear_btn = ctk.CTkButton(self._draw_tab, text="Clear Canvas", command=self._on_dtab_clear_click, 
                                         height=self.BTN_HEIGHT, font=self.BTN_FONT)
        
        # Uses different cmd to upload scan btn bc it must read canvas instead of img
        self._scan_btn_d_tab = ctk.CTkButton(self._draw_tab, text="Scan Drawing", command=self._on_dtab_scan_click,
                                             height=self.BTN_HEIGHT, font=self.BTN_FONT) 


    def _create_upload_tab(self):

        self._upload_tab = self._inputs_tabs.add("Upload")

        # Usr buttons
        self._upload_btn = ctk.CTkButton(self._upload_tab, text="Upload Image", command=self._on_utab_upload_click, 
                                         height=self.BTN_HEIGHT, font=self.BTN_FONT)
        self._scan_btn_u_tab = ctk.CTkButton(self._upload_tab, text="Scan Image", command=self._on_utab_scan_click,
                                             height=self.BTN_HEIGHT, font=self.BTN_FONT)

        # img
        self._usr_img_lbl = ctk.CTkLabel(self._upload_tab, text="")

        # Now fix size of ghost img
        self._pil_canvas_cpy = Image.new("L", (500, 500), 255) # 255 = White
        print(self._pil_canvas_cpy.size)
        self._pil_canvas_cpy_draw = ImageDraw.Draw(self._pil_canvas_cpy)

    def _create_results_frame(self):

        # So can be stuck into right column
        self._results_frame = ctk.CTkFrame(self)
        
        # TODO: add gpu lbl here

        # net outputs
        # Fill with values first to not have 'CTkComboBox' as the default value
        self._net_selector = ctk.CTkComboBox(self, values=self.controller.get_pretrained_networks())
        self._net_pred_lbl = ctk.CTkLabel(self, text="The network's prediction will be displayed here.")
        self._net_conf_lbl = ctk.CTkLabel(self, text="The network's confidence will be displayed here.")


    def _layout_net_tab(self):
        pass


    def _layout_draw_tab(self):
        
        self._draw_tab_txt_box.pack(fill='x', padx=3.5, pady=5, side='top')

        # Drawing canvas
        self._canvas.pack(fill = 'both', expand=True, padx=5, pady=5)
        self._canvas.configure(width=self._canvas.winfo_height())

        # Btns
        self._clear_btn.pack(fill = 'both', expand=True, padx=2.5, pady=5, side='left')
        self._scan_btn_d_tab.pack(fill = 'both', expand=True, padx=2.5, pady=5, side='right')

        self._canvas.update()



    def _layout_upload_tab(self):
        
        # img
        self._usr_img_lbl.pack(fill = 'both', expand=True, padx=5, pady=5)

        # btns
        self._upload_btn.pack(fill = 'both', expand=True, padx=5, pady=5, side='left')
        self._scan_btn_u_tab.pack(fill = 'both', expand=True, padx=5, pady=5, side='right')


    def _layout_results_frame(self):
        return

        # Rejsults stuff is column 1
        self._results_frame.grid(row=0, column=1, padx=20, pady=20, sticky='nesw')

        # net outputs
        self._net_selector.grid(row=2, column=1, padx=20, pady=20)
        self._net_pred_lbl.grid(row=3, column=1, padx=20, pady=5)
        self._net_conf_lbl.grid(row=4, column=1, padx=20, pady=5)


    def _canvas_draw(self, event):

        # x, y of mouse position when in motion
        x, y = event.x, event.y

        # Coords of rectangle containing brush circle
        # Separate because PIL and tkinter reverse the order
        x1, y1 = x-self.BRUSH_SIZE/2, y+self.BRUSH_SIZE/2
        x2, y2 = x+self.BRUSH_SIZE/2, y-self.BRUSH_SIZE/2

        # Symmetrical oval is just circle
        self._canvas.create_oval((x1, y1), (x2, y2), fill='black')  

        # Mirror the changes on the PIL copy
        # TODO: use relative coordinatess
        # self._pil_canvas_cpy_draw.ellipse(((x1, y2), (x2, y1)), fill='black', width=5)

    def _on_dtab_clear_click(self):
        
        # Clear the canvas
        self._canvas.delete("all")


    def _on_dtab_scan_click(self):
        '''Scan the drawing in self._canvas.'''

        self._capture_canvas()

        pass


    def _on_utab_upload_click(self):

        # temp for easier testing
        if ScannerFrame.SKIP_IMG_SELECT:
            img_path = r"C:\Users\ikram\Documents\Schoolwork\compsci\test_img.png"
        else:
            img_path = ctk.filedialog.askopenfilename(title="Open Image File", 
                                          filetypes=[("Image files", "*.png *.jpg *.jpeg *.gif *.bmp *.ico")])

        if not img_path: return

        self._curr_img_pil =  Image.open(img_path) 
        self._curr_img_ctk = ctk.CTkImage(self._curr_img_pil, size=ScannerFrame.USR_IMG_SIZE)

        self._usr_img_lbl.configure(image=self._curr_img_ctk)


    def _on_utab_scan_click(self):
        
        if not self._curr_img_pil:
            return

        img_arr = self._conv_img_to_arr()

        # TODO: network outputs
        self.controller.network.fwd(img_arr)


    def _capture_canvas(self) -> Image.Image:
        x0 = self._canvas.winfo_rootx()
        y0 = self._canvas.winfo_rooty()
        x1 = x0 + self._canvas.winfo_width()
        y1 = y0 + self._canvas.winfo_height()

        img = ImageGrab.grab((x0, y0, x1, y1))
        img.show() # for debugging
        return img




    def _conv_img_to_arr(self) -> cp.ndarray:
            
        img_grey_resized = self._curr_img_pil.resize(self.controller.DIGIT_IMG_SIZE).convert(mode='F')
        img_arr = cp.asarray(img_grey_resized) / 255

        return img_arr



    