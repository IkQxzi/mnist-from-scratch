
# internal
from typing import TYPE_CHECKING

# 3rd party

# For maths & NN stuff
import cupy as cp
import numpy as np

# GUI stuff
import customtkinter as ctk
import tkinter as tk # For canvas, ctkcanvas is bad
from PIL import Image, ImageDraw, ImageGrab, ImageOps
import matplotlib.axes, matplotlib.figure, matplotlib.container
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# local
if TYPE_CHECKING:
    from app_ui.controller import App

# For readability
cp.set_printoptions(suppress=True, precision=3)

class ScannerFrame(ctk.CTkFrame):
    
    # Constants
    USR_IMG_SIZE = (500, 500)
    SKIP_IMG_SELECT = False

    # Draw tab constants
    BRUSH_SIZE = 75
    CANVAS_SIDELENGTH = 800

    # Multi-use attributes
    controller: "App"
    _curr_img_pil: Image.Image
    _curr_img_arr: cp.ndarray # 28x28
    _is_img_displayed: bool

    # Tabs and Frames
    _inputs_tabs: ctk.CTkTabview
    _net_tab: ctk.CTkFrame
    _draw_tab: ctk.CTkFrame
    _upload_tab: ctk.CTkFrame
    _results_frame: ctk.CTkFrame
    
    # Network tab widgets
    _active_net_name_lbl: ctk.CTkLabel
    _active_net_acc_lbl: ctk.CTkLabel
    _active_net_loss_lbl: ctk.CTkLabel
    _active_net_size_lbl: ctk.CTkLabel
    
    _net_select_title_lbl: ctk.CTkLabel
    _net_dropdown: ctk.CTkComboBox
    _selected_net_name_lbl: ctk.CTkLabel
    _selected_net_acc_lbl: ctk.CTkLabel
    _selected_net_loss_lbl: ctk.CTkLabel
    _selected_net_size_lbl: ctk.CTkLabel
    _net_select_btn: ctk.CTkButton
    _net_selected_popup_lbl: ctk.CTkLabel
    
    # Draw tab widgets
    _draw_tab_txt_box: ctk.CTkLabel
    _canvas: tk.Canvas
    _pil_canvas_cpy: Image.Image
    _pil_canvas_cpy_draw: ImageDraw.ImageDraw
    _clear_btn: ctk.CTkButton
    _scan_btn_d_tab: ctk.CTkButton

    # Upload tab wijdgets
    _upload_btn: ctk.CTkButton
    _scan_btn_u_tab: ctk.CTkButton
    _usr_img_lbl: ctk.CTkLabel
    _usr_img_pil: Image.Image
    _usr_img_ctk: ctk.CTkImage
    
    # Results widgets
    _net_pred_txt_lbl: ctk.CTkLabel
    _net_pred_num_lbl: ctk.CTkLabel
    _net_conf_txt_lbl: ctk.CTkLabel
    _net_conf_num_lbl: ctk.CTkLabel
    _ax: matplotlib.axes.Axes
    _fig: matplotlib.figure.Figure
    _bars: matplotlib.container.BarContainer
    _conf_figure_canvas: FigureCanvasTkAgg
    

    def __init__(self, container, controller: "App"):
        
        # Put frame inside of container
        super().__init__(master=container)

        self.controller = controller


        # Input side split into tabs
        self._inputs_tabs = ctk.CTkTabview(self)
        self._create_net_tab()
        self._create_draw_tab()
        self._create_upload_tab()
        self._create_results_frame()
        
        # Split the frame into jtwo equal columns & 1 row
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Split layout in half
        self._inputs_tabs.grid(row=0, column=0, sticky='nsew')
        self._results_frame.grid(row=0, column=1, sticky='nsew')

        # Individual layouts
        self._layout_net_tab()
        self._layout_draw_tab()
        self._layout_upload_tab()
        self._layout_results_frame()

    # -----------------------
    # Public Methods
    # -----------------------

    def reset_frame(self):
        # Remove the image and stuff
        pass

    # -----------------------
    # Creation
    # -----------------------

    def _create_net_tab(self):

        # Add net tab to the tab list
        self._net_tab = self._inputs_tabs.add("Network")

        # Name of active network (by default none selected)
        self._active_net_name_lbl = ctk.CTkLabel(self._net_tab, text="Active Network: NONE ACTIVE",
                                                 fg_color=self.controller.PRIMARY_LBL_COLOUR,
                                                 corner_radius=self.controller.LBL_CORNER_RADIUS,
                                                 font=self.controller.BIG_BTN_FONT)
        
        # Metrics of active network 
        self._active_net_acc_lbl = ctk.CTkLabel(self._net_tab, text="Active Network Accuracy: NONE ACTIVE",
                                                 fg_color=self.controller.PRIMARY_LBL_COLOUR,
                                                 corner_radius=self.controller.LBL_CORNER_RADIUS,
                                                 font=self.controller.BTN_FONT)
        
        self._active_net_loss_lbl = ctk.CTkLabel(self._net_tab, text="Active Network Loss: NONE ACTIVE",
                                                 fg_color=self.controller.PRIMARY_LBL_COLOUR,
                                                 corner_radius=self.controller.LBL_CORNER_RADIUS,
                                                 font=self.controller.BTN_FONT)
        
        self._active_net_size_lbl = ctk.CTkLabel(self._net_tab, text="Active Network Size: NONE ACTIVE",
                                                 fg_color=self.controller.PRIMARY_LBL_COLOUR,
                                                 corner_radius=self.controller.LBL_CORNER_RADIUS,
                                                 font=self.controller.BTN_FONT)
        
        # To separate active nektwork section from network selection
        self._net_select_title_lbl = ctk.CTkLabel(self._net_tab, text='Available Networks',
                                                 fg_color=self.controller.PRIMARY_LBL_COLOUR,
                                                 corner_radius=self.controller.LBL_CORNER_RADIUS,
                                                 font=self.controller.BIGGEST_BTN_FONT)

        # List of available networks 
        # Fill with values first to not have 'CTkComboBox' as the default value
        # Contains name of selected network at the top
        self._net_dropdown = ctk.CTkComboBox(self._net_tab, values=self.controller.net_infos['net_names'],
                                             font=self.controller.BIG_BTN_FONT,
                                             dropdown_font=self.controller.BIG_BTN_FONT,
                                             command=self._on_net_dropdown_update) # Cur val passed in automatically


        # Get the metrics for the network which will be at the top of the combobox on startup
        startup_net_acc = self.controller.net_infos['net_accs'][0]
        startup_net_loss = self.controller.net_infos['net_losses'][0]
        startup_net_size = self.controller.net_infos['net_sizes'][0]


        # Metrics of selected network
        self._selected_net_acc_lbl = ctk.CTkLabel(self._net_tab, text=f"Accuracy: {startup_net_acc}",
                                                 fg_color=self.controller.PRIMARY_LBL_COLOUR,
                                                 corner_radius=self.controller.LBL_CORNER_RADIUS,
                                                 font=self.controller.BTN_FONT)
        
        self._selected_net_loss_lbl = ctk.CTkLabel(self._net_tab, text=f"Loss: {startup_net_loss}",
                                                 fg_color=self.controller.PRIMARY_LBL_COLOUR,
                                                 corner_radius=self.controller.LBL_CORNER_RADIUS,
                                                 font=self.controller.BTN_FONT)
        
        self._selected_net_size_lbl = ctk.CTkLabel(self._net_tab, text=f"Size: {startup_net_size}",
                                                 fg_color=self.controller.PRIMARY_LBL_COLOUR,
                                                 corner_radius=self.controller.LBL_CORNER_RADIUS,
                                                 font=self.controller.BTN_FONT)
        
        # Confirmation button so that the user can look through networks without loading each one
        self._net_select_btn = ctk.CTkButton(self._net_tab, text="Select Network", command=self._on_net_select_btn_click,
                                             font=self.controller.BIG_BTN_FONT)
        
        # Will only be shown when required in layout
        self._net_selected_popup_lbl = ctk.CTkLabel(self._net_tab, text="Network selected!",
                                                 corner_radius=self.controller.LBL_CORNER_RADIUS,
                                                 font=self.controller.BTN_FONT, text_color='#2B2B2B')


    def _create_draw_tab(self):
        
        # Add to input tabs
        self._draw_tab = self._inputs_tabs.add("Draw")

        self._draw_tab_txt_box = ctk.CTkLabel(self._draw_tab, text="Try to fill the canvas as much as possible for the best network predictions.",
                                             height=self.controller.SMALL_BTN_HEIGHT, font=self.controller.SMALL_BTN_FONT, 
                                             corner_radius=self.controller.LBL_CORNER_RADIUS, fg_color=self.controller.PRIMARY_BTN_COLOUR,) 

        # For user to draw on 
        self._canvas = tk.Canvas(self._draw_tab, bg='white', 
                                     height=self.CANVAS_SIDELENGTH,
                                     width=self.CANVAS_SIDELENGTH)

        # We must create a ghost copy of the canvas to avoid having to work with file exports
        # For now, use (1, 1) size; match to canvas in layout method
        self._pil_canvas_cpy: Image.Image
        self._pil_canvas_cpy_draw: ImageDraw.ImageDraw # Must be instantiated last
        
        # Moving the mouse should only draw when B1 (LMB) is held down
        self._canvas.bind('<B1-Motion>', self._canvas_draw)

        self._clear_btn = ctk.CTkButton(self._draw_tab, text="Clear Canvas", command=self._on_d_tab_clear_click, 
                                         height=self.controller.BTN_HEIGHT, font=self.controller.BTN_FONT)
        
        # Uses different cmd to upload scan btn bc it must read canvas instead of img
        self._scan_btn_d_tab = ctk.CTkButton(self._draw_tab, text="Scan Drawing", command=self._on_d_tab_scan_click,
                                             height=self.controller.BTN_HEIGHT, font=self.controller.BTN_FONT) 


    def _create_upload_tab(self):

        self._upload_tab = self._inputs_tabs.add("Upload")

        # Usr buttons
        self._upload_btn = ctk.CTkButton(self._upload_tab, text="Upload Image", command=self._on_u_tab_upload_click, 
                                         height=self.controller.BTN_HEIGHT, font=self.controller.BTN_FONT)
        self._scan_btn_u_tab = ctk.CTkButton(self._upload_tab, text="Scan Image", command=self._on_u_tab_scan_click,
                                             height=self.controller.BTN_HEIGHT, font=self.controller.BTN_FONT)

        # img
        self._usr_img_lbl = ctk.CTkLabel(self._upload_tab, text="")

        # Now fix size of ghost img
        self._pil_canvas_cpy = Image.new("L", (500, 500), 255) # 255 = White
        print(self._pil_canvas_cpy.size)
        self._pil_canvas_cpy_draw = ImageDraw.Draw(self._pil_canvas_cpy)


    def _create_results_frame(self):

        # So can be stuck into right column
        self._results_frame = ctk.CTkFrame(self, border_width=5)
        
        self._active_net_lbl_r_frame = ctk.CTkLabel(self._results_frame,
                                                    text="Active Network: NONE",
                                                    font=self.controller.BIGGEST_BTN_FONT,
                                                    fg_color=self.controller.PRIMARY_LBL_COLOUR,
                                                    corner_radius=self.controller.LBL_CORNER_RADIUS)

        # Prediction labels
        self._net_pred_txt_lbl = ctk.CTkLabel(self._results_frame, 
                                              text="Network Prediction:",
                                              font=self.controller.BIG_BTN_FONT)
        
        self._net_pred_num_lbl = ctk.CTkLabel(self._results_frame, text='N/A',
                                              fg_color=self.controller.PRIMARY_LBL_COLOUR, 
                                              font=('Arial', 60, 'bold'))


        # Main confidence labels
        self._net_conf_txt_lbl = ctk.CTkLabel(self._results_frame, 
                                              text="Confidence:",
                                              font=self.controller.BIG_BTN_FONT)
        
        self._net_conf_num_lbl = ctk.CTkLabel(self._results_frame, text="N/A",
                                              fg_color=self.controller.PRIMARY_LBL_COLOUR,
                                              font=self.controller.BIGGEST_BTN_FONT)

        self._conf_fig_title = ctk.CTkLabel(self._results_frame, text='Prediction Distribution',
                                           font=self.controller.BIGGEST_BTN_FONT,
                                           fg_color=self.controller.PRIMARY_LBL_COLOUR,
                                           corner_radius=self.controller.LBL_CORNER_RADIUS)

        # Initialise confidence score figure,
        # creating the attributes below
        self._setup_conf_figure()
        self._ax: matplotlib.axes.Axes
        self._fig: matplotlib.figure.Figure
        self._bars: matplotlib.container.BarContainer
        self._conf_figure_canvas: FigureCanvasTkAgg
    

    # -----------------------
    # Layout
    # -----------------------

    def _layout_net_tab(self):

        # Local constants
        COLUMN_COUNT = 2
        ROW_COUNT = 8

        # Force the number of columns specified
        for i in range(COLUMN_COUNT):
            self._net_tab.grid_columnconfigure(index=i, weight=1)
        
        # Force the number of rows specified
        for i in range(ROW_COUNT):
            self._net_tab.grid_rowconfigure(index=i, weight=1)

        
        # Place active net name on the left middle 
        # And span 3 rows so it is the same as the metrics table on the right
        self._active_net_name_lbl.grid(row=1, column=0, padx=5, pady=50,
                                       )
        
        # Vertically stack active net metrics on the right side
        self._active_net_acc_lbl.grid(row=0, column=1, padx=5, pady=15)
        self._active_net_loss_lbl.grid(row=1, column=1, padx=5, pady=15)
        self._active_net_size_lbl.grid(row=2, column=1, padx=5, pady=15)

        # Place network selection title in the section below active network info
        # And make it span the entire tabk
        self._net_select_title_lbl.grid(row=3, sticky='ew', columnspan=2, pady=50)

        # Place net dropdown left of metrics (aligned with middle row)
        self._net_dropdown.grid(row=5, column=0, padx=5, pady=5, sticky='ew')

        # PLace selected net metrics bottom right
        self._selected_net_acc_lbl.grid(row=4, column=1, padx=5, pady=15)
        self._selected_net_loss_lbl.grid(row=5, column=1, padx=5, pady=15)
        self._selected_net_size_lbl.grid(row=6, column=1, padx=5, pady=15)

        # Place network selection button bottom middle, across both columns
        self._net_select_btn.grid(row=7, padx=5, pady=(30, 0), sticky='ew', columnspan=2)

        # Place network selected popup (disabled)
        self._net_selected_popup_lbl.configure(state='disabled')
        self._net_selected_popup_lbl.grid(row=8, padx=10, pady=5, sticky='ew', columnspan=2)


    def _layout_draw_tab(self):
        
        self._draw_tab_txt_box.pack(fill='x', padx=3.5, pady=5, side='top')

        # Drawing canvas, do not fill space otherwise it will not be square
        self._canvas.pack(expand=True, padx=5, pady=5)
        # self._canvas.configure(width=self._canvas.winfo_height())

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

        # Split results frame vertically in half
        self._results_frame.columnconfigure(index=[0, 1], weight=1)
        
        # Have the row for active network and confidence titles smaller
        # And other rows all equal
        self._results_frame.rowconfigure(index=[0, 3], weight=1)
        self._results_frame.rowconfigure(index=[1, 2, 4], weight=3)

        self._active_net_lbl_r_frame.grid(row=0, columnspan=2, padx=15, pady=15)

        # Place prediction text top left
        self._net_pred_txt_lbl.grid(row=1, column=0, sticky='nse', padx=15)
    
        # Place prediciton number right of prediction text
        self._net_pred_num_lbl.grid(row=1, column=1, sticky='nsew')

        # Place confidence text below prediction info
        self._net_conf_txt_lbl.grid(row=2, column=0, sticky='nse', padx=15)

        # Place prediction output right of prediction text
        self._net_conf_num_lbl.grid(row=2, column=1, sticky='nsew')

        # Put title underneath the above
        self._conf_fig_title.grid(row=3, columnspan=2)

        # Place confidence graph below all the above, spanning both columns
        self._conf_figure_canvas.draw() # Acts like a canvas, must be drawn first
        self._conf_figure_canvas.get_tk_widget().grid(row=4, columnspan=2, sticky='nsew', padx=15, pady=15)


    # -----------------------
    # Callback
    # -----------------------

    # Note that tkinter automatically passes in the selected net name when called
    def _on_net_dropdown_update(self, selected_net_name: str) -> None:
        '''Update the displayed metrics for the selected network when a new one is selected
        and disable the network selected confirmation dialogue.'''

        # If popup is present then disable confirmation dialogue to prevent confusing user
        if self._net_selected_popup_lbl.winfo_viewable():
            self._net_selected_popup_lbl.configure(state='disabled')

        # Get the index of net within controller net infos dict to get metrics later
        net_index = self.controller.net_infos['net_names'].index(selected_net_name)

        # Get the new values for acc, loss, size from controller
        avg_acc = self.controller.net_infos['net_accs'][net_index]
        avg_loss = self.controller.net_infos['net_losses'][net_index]
        size = self.controller.net_infos['net_sizes'][net_index]
        
        # Update the metrics labels with the new values
        self._selected_net_acc_lbl.configure(text=f'Accuracy: {float(avg_acc):.1f}%')
        self._selected_net_loss_lbl.configure(text=f'Loss: {float(avg_loss):.2f}')
        self._selected_net_size_lbl.configure(text=f'Size: {size} layers')


    def _on_net_select_btn_click(self) -> None:
        '''Change the active network to the one in self._net_dropdown and display a popup.'''

        # Get the currently selected network in dropdown
        net_name = self._net_dropdown.get()
        print(net_name)

        # Get the index of net within controller net infos dict to get metrics later
        net_index = self.controller.net_infos['net_names'].index(net_name)
        
        # Set as active network
        self.controller.set_network(net_name)

        # Get the new values for acc, loss, size from controller
        avg_acc = self.controller.net_infos['net_accs'][net_index]
        avg_loss = self.controller.net_infos['net_losses'][net_index]
        size = self.controller.net_infos['net_sizes'][net_index]

        # Update the active network name label network tab
        self._active_net_name_lbl.configure(text=f'Active Network: {net_name}')
        
        # Do same for results frame
        self._active_net_lbl_r_frame.configure(text=f'Active Network: {net_name}')

        # Update the metrics labels with the new values
        self._active_net_acc_lbl.configure(text=f'Accuracy: {float(avg_acc):.1f}%')
        self._active_net_loss_lbl.configure(text=f'Loss: {float(avg_loss):.2f}')
        self._active_net_size_lbl.configure(text=f'Size: {size} layers')

        # Show popup telling user network is selected
        # Will be reset when another selection is made
        self._net_selected_popup_lbl.configure(state='active')


    def _on_d_tab_clear_click(self):
        
        # Clear the canvas
        self._canvas.delete("all")


    def _on_d_tab_scan_click(self):
        '''Scan the drawing in self._canvas.'''

        # Scan the image from user
        usr_draw = self._capture_canvas()

        # Update self._curr_img_arr
        self._set_formatted_img(usr_draw)

        # Use attribute to fwd and update
        self._fwd_and_update_results_frame(self._curr_img_arr)

        print(f'probs: {self.controller.network.probs}')
        print(f'conf_scores: {self.controller.network.conf_scores}')


    def _on_u_tab_upload_click(self):

        # temp for easier testing
        if ScannerFrame.SKIP_IMG_SELECT:
            img_path = r"C:\Users\ikram\Documents\Schoolwork\compsci\test_img.png"
        else:
            img_path = ctk.filedialog.askopenfilename(title="Open Image File", 
                                          filetypes=[("Image files", "*.png *.jpg *.jpeg *.gif *.bmp *.ico")], 
                                          initialdir='C://testdir')

        if not img_path: return

        self._usr_img_pil = Image.open(img_path) 
        self._usr_img_ctk = ctk.CTkImage(self._usr_img_pil, size=ScannerFrame.USR_IMG_SIZE)

        self._usr_img_lbl.configure(image=self._usr_img_ctk)


    def _on_u_tab_scan_click(self):

        # Do nothing if user image not displayed        
        if not self._usr_img_pil:
            return

        # Process usr img into self._curr_img_arr
        self._set_formatted_img(self._usr_img_pil)

        # Use curr img to update results frame
        self._fwd_and_update_results_frame(self._curr_img_arr)


    # -----------------------
    # Functional methods
    # -----------------------

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


    def _capture_canvas(self) -> Image.Image:
        '''Get and return the raw Image from canvas.'''

        # Get coords of each corner of canvas
        x0 = self._canvas.winfo_rootx()
        y0 = self._canvas.winfo_rooty()
        x1 = x0 + self._canvas.winfo_width()
        y1 = y0 + self._canvas.winfo_height()

        # Grab usr drawing, set it as attrib
        usr_draw = ImageGrab.grab((x0, y0, x1, y1))
        
        # For debugging
        # usr_draw.show()
        return usr_draw


    def _set_formatted_img(self, img: Image.Image):
        '''
        Resize, conv to cp arr, and invert the given PIL image.
        Set the img and its cp arr as attributes.
        '''

        # Greyscale, invert bc MNIST expects white on black
        img = img.convert(mode='L')
        img = ImageOps.invert(img)

        # The bounding box of the digit
        digit_bbox = img.getbbox()

        # If no digit, forward an array of nothing
        if digit_bbox is None:
            self._curr_img_arr = cp.zeros((1, 784), dtype='float32')
            return

        # Crop image so digit just touches edge
        img_cropped = img.crop(digit_bbox)

        # Will be useful for padding
        width, height = img_cropped.size
        aspect_ratio = width / height

        # To force digit to be 20x20 px
        if width > height:
            new_width = 20
            new_height = max(1, int(20 / aspect_ratio))
        else: 
            new_height = 20
            new_width = max(1, int(20 * aspect_ratio))

        # Resize image and use LANCZOS antialiasing
        img_resized = img_cropped.resize(size=(new_width, new_height), resample=Image.Resampling.LANCZOS)

        # Conv img to 2d arr
        img_2d_arr = np.asarray(img_resized, dtype='float32')
        total_img_mass = np.sum(img_2d_arr)

        # Get the x & y coordinates of every px
        y_indices, x_indices = np.indices(img_2d_arr.shape)
        
        # Find centre of mass (weighted avg.) as done in MNIST
        cor_y = np.sum(y_indices * img_2d_arr) / total_img_mass
        cor_x = np.sum(x_indices * img_2d_arr) / total_img_mass
        
        # Paste digit in centre of blank img
        paste_x = int(round(14.0 - cor_x))
        paste_y = int(round(14.0 - cor_y))

        # Black 28x28 image to pad digit 4px all around
        padded_img = Image.new('L', (28, 28), color=0)

        # Place img in centre of padded img
        padded_img.paste(im=img_resized, box=(paste_x, paste_y))
        
        # For debugging
        # padded_img.show() 

        # Now conv padded img to cupy arr
        padded_img_2d_arr = cp.asarray(padded_img, dtype='float32') # float32 for GPU
        
        # Fix contrast issues from imperfect whites
        max_val = cp.max(padded_img_2d_arr)
        if max_val > 0:
            padded_img_2d_arr /= max_val
        else:
            padded_img_2d_arr /= 255.0 # Fallback option
        
        print(f'inputs:{padded_img_2d_arr}')
        with open('net_outputs.txt', 'w') as file:
            file.write(cp.array2string(padded_img_2d_arr))
        
        # Reshape to 2d arr of (1, 784) for fwding
        print(f'2d arr: {padded_img_2d_arr.shape}')
        self._curr_img_arr = padded_img_2d_arr.reshape(1, 784)
        print(self._curr_img_arr.shape)


    def _setup_conf_figure(self):
        '''Setup the confidence scores figure and attach to the results frame.'''

        # White looks best with ctk
        grid_colour = 'white'

        # Create figure object, 5x3in, high dpi to be sharp
        self._fig = Figure(figsize=(5, 3), dpi=100)

        # Set the background colour to be the same as the rest of the app
        self._fig.patch.set_facecolor(self.controller.PRIMARY_BG_COLOUR)

        # Add axes (mplib Axis obj)
        self._ax = self._fig.add_subplot()

        # Add labels to axes and set them to white
        self._ax.set_ylabel('Confidence')
        self._ax.set_xlabel('Digit')
        self._ax.yaxis.label.set_color(grid_colour)
        self._ax.xaxis.label.set_color(grid_colour)

        # Change gridlines to be white
        self._ax.tick_params(colors=grid_colour)
        
        # Hide the top and right borders
        self._ax.spines['top'].set_visible(False)
        self._ax.spines['right'].set_visible(False)
        
        # Set other borders to white
        self._ax.spines['left'].set_color(grid_colour)
        self._ax.spines['bottom'].set_color(grid_colour)

        digit_strs = [str(i) for i in range(10)]
        initial_conf_scores = [0 for _ in range(10)]

        # Save as attribute to modify height later
        self._bars = self._ax.bar(x=digit_strs, height=initial_conf_scores,
                                 color=self.controller.SECONDARY_BLUE_COLOUR)
        
        # Make y-axis go from 0 to 100 (as conf scores do)
        self._ax.set_ylim(bottom=0, top=100)

        # Create canvas with graph embedded
        self._conf_figure_canvas = FigureCanvasTkAgg(self._fig, master=self._results_frame)


    def _update_conf_figure(self, conf_scores_gpu: cp.ndarray, pred_digit: int):
        '''Update the confidence figure in the results frame,
        using the confidence scores provided directly from the GPU,
        and the main predicted digit.'''

        # Move conf scores to CPU after flattening to 1D
        conf_scores_cpu = cp.asnumpy(conf_scores_gpu.flatten())

        # Update bars
        for i, bar in enumerate(self._bars):
            
            # Update bar heights
            bar.set_height(conf_scores_cpu[i])

            # Highlight main prediction bar in green
            if i == pred_digit:
                bar.set_color(self.controller.PRIMARY_GREEN_COLOUR)
            
            # Important to reset the colours
            else: 
                bar.set_color(self.controller.SECONDARY_BLUE_COLOUR)

        # Apply changes
        self._conf_figure_canvas.draw()

    
    def _fwd_and_update_results_frame(self, inputs: cp.ndarray):
        '''Forward inputs to network and update all aspects of the results frame.'''

        # Forward inputs to net and get outputs
        self.controller.network.fwd(inputs)
        pred_digit, main_conf_score = self.controller.network.get_pred_digit_and_conf()

        # Update main prediction
        self._net_pred_num_lbl.configure(text=f'{pred_digit}')

        # Update conf score and format to 1 d.p.
        self._net_conf_num_lbl.configure(text=f'{main_conf_score:.1f}%')

        # Update figure with conf score from gpu and pred digit
        self._update_conf_figure(conf_scores_gpu=self.controller.network.conf_scores, 
                                 pred_digit=pred_digit)