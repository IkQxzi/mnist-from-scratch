
# internal
from typing import TYPE_CHECKING

# 3rd party
import cupy as cp
import customtkinter as ctk
from PIL import Image, ImageDraw, ImageGrab, ImageOps
import matplotlib.axes, matplotlib.figure, matplotlib.container
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# local
if TYPE_CHECKING:
    from app_ui.controller import App

class ScannerFrame(ctk.CTkFrame):
    
    # Constants
    USR_IMG_SIZE = (500, 500)
    SKIP_IMG_SELECT = False
    BRUSH_SIZE = 125

    # Button sizes
    BTN_HEIGHT = 50
    SMALL_BTN_HEIGHT = 20

    # Fonts
    SMALL_BTN_FONT = ("Arial", 14, "bold")
    BTN_FONT = ("Arial", 25, "bold")
    BIG_BTN_FONT = ("Arial", 35, "bold")
    BIGGEST_BTN_FONT= ("Arial", 45, "bold")

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
    _canvas: ctk.CTkCanvas
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
                                                 font=self.BIG_BTN_FONT)
        
        # Metrics of active network 
        self._active_net_acc_lbl = ctk.CTkLabel(self._net_tab, text="Active Network Accuracy: NONE ACTIVE",
                                                 fg_color=self.controller.PRIMARY_LBL_COLOUR,
                                                 corner_radius=self.controller.LBL_CORNER_RADIUS,
                                                 font=self.BTN_FONT)
        
        self._active_net_loss_lbl = ctk.CTkLabel(self._net_tab, text="Active Network Loss: NONE ACTIVE",
                                                 fg_color=self.controller.PRIMARY_LBL_COLOUR,
                                                 corner_radius=self.controller.LBL_CORNER_RADIUS,
                                                 font=self.BTN_FONT)
        
        self._active_net_size_lbl = ctk.CTkLabel(self._net_tab, text="Active Network Size: NONE ACTIVE",
                                                 fg_color=self.controller.PRIMARY_LBL_COLOUR,
                                                 corner_radius=self.controller.LBL_CORNER_RADIUS,
                                                 font=self.BTN_FONT)
        
        # To separate active nektwork section from network selection
        self._net_select_title_lbl = ctk.CTkLabel(self._net_tab, text='Available Networks',
                                                 fg_color=self.controller.PRIMARY_LBL_COLOUR,
                                                 corner_radius=self.controller.LBL_CORNER_RADIUS,
                                                 font=self.BIGGEST_BTN_FONT)

        # List of available networks 
        # Fill with values first to not have 'CTkComboBox' as the default value
        # Contains name of selected network at the top
        self._net_dropdown = ctk.CTkComboBox(self._net_tab, values=self.controller.net_infos['net_names'],
                                             font=self.BIG_BTN_FONT,
                                             dropdown_font=self.BIG_BTN_FONT,
                                             command=self._on_net_dropdown_update) # Cur val passed in automatically


        # Get the metrics for the network which will be at the top of the combobox on startup
        startup_net_acc = self.controller.net_infos['net_accs'][0]
        startup_net_loss = self.controller.net_infos['net_losses'][0]
        startup_net_size = self.controller.net_infos['net_sizes'][0]


        # Metrics of selected network
        self._selected_net_acc_lbl = ctk.CTkLabel(self._net_tab, text=f"Accuracy: {startup_net_acc}",
                                                 fg_color=self.controller.PRIMARY_LBL_COLOUR,
                                                 corner_radius=self.controller.LBL_CORNER_RADIUS,
                                                 font=self.BTN_FONT)
        
        self._selected_net_loss_lbl = ctk.CTkLabel(self._net_tab, text=f"Loss: {startup_net_loss}",
                                                 fg_color=self.controller.PRIMARY_LBL_COLOUR,
                                                 corner_radius=self.controller.LBL_CORNER_RADIUS,
                                                 font=self.BTN_FONT)
        
        self._selected_net_size_lbl = ctk.CTkLabel(self._net_tab, text=f"Size: {startup_net_size}",
                                                 fg_color=self.controller.PRIMARY_LBL_COLOUR,
                                                 corner_radius=self.controller.LBL_CORNER_RADIUS,
                                                 font=self.BTN_FONT)
        
        # Confirmation button so that the user can look through networks without loading each one
        self._net_select_btn = ctk.CTkButton(self._net_tab, text="Select Network", command=self._on_net_select_btn_click,
                                             font=self.BIG_BTN_FONT)
        
        # Will only be shown when required in layout
        self._net_selected_popup_lbl = ctk.CTkLabel(self._net_tab, text="Network selected!",
                                                 corner_radius=self.controller.LBL_CORNER_RADIUS,
                                                 font=self.BTN_FONT, text_color='#2B2B2B')


    def _create_draw_tab(self):
        
        # Add to input tabs
        self._draw_tab = self._inputs_tabs.add("Draw")

        self._draw_tab_txt_box = ctk.CTkLabel(self._draw_tab, text="Try to fill the canvas as much as possible for the best network predictions.",
                                             height=self.SMALL_BTN_HEIGHT, font=self.SMALL_BTN_FONT, 
                                             corner_radius=self.controller.LBL_CORNER_RADIUS, fg_color=self.controller.PRIMARY_BTN_COLOUR,) 

        # For user to draw on 
        self._canvas = ctk.CTkCanvas(self._draw_tab)

        # We must create a ghost copy of the canvas to avoid having to work with file exports
        # For now, use (1, 1) size; match to canvas in layout method
        self._pil_canvas_cpy: Image.Image
        self._pil_canvas_cpy_draw: ImageDraw.ImageDraw # Must be instantiated last
        
        # Moving the mouse should only draw when B1 (LMB) is held down
        self._canvas.bind('<B1-Motion>', self._canvas_draw)

        self._clear_btn = ctk.CTkButton(self._draw_tab, text="Clear Canvas", command=self._on_d_tab_clear_click, 
                                         height=self.BTN_HEIGHT, font=self.BTN_FONT)
        
        # Uses different cmd to upload scan btn bc it must read canvas instead of img
        self._scan_btn_d_tab = ctk.CTkButton(self._draw_tab, text="Scan Drawing", command=self._on_d_tab_scan_click,
                                             height=self.BTN_HEIGHT, font=self.BTN_FONT) 


    def _create_upload_tab(self):

        self._upload_tab = self._inputs_tabs.add("Upload")

        # Usr buttons
        self._upload_btn = ctk.CTkButton(self._upload_tab, text="Upload Image", command=self._on_u_tab_upload_click, 
                                         height=self.BTN_HEIGHT, font=self.BTN_FONT)
        self._scan_btn_u_tab = ctk.CTkButton(self._upload_tab, text="Scan Image", command=self._on_u_tab_scan_click,
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

        # Prediction labels
        self._net_pred_txt_lbl = ctk.CTkLabel(self._results_frame, 
                                              text="The network's prediction will be displayed here.")
        self._net_pred_num_lbl = ctk.CTkLabel(self._results_frame, text='X', 
                                              height=100, width=100,
                                              fg_color=self.controller.PRIMARY_LBL_COLOUR)


        # Main confidence labels
        self._net_conf_txt_lbl = ctk.CTkLabel(self._results_frame, 
                                              text="The network's confidence will be displayed here.")
        self._net_conf_num_lbl = ctk.CTkLabel(self._results_frame, text="XX%")

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
        self._active_net_name_lbl.grid(row=1, column=0, padx=100, pady=50,
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

        # Split results frame in half
        self._results_frame.columnconfigure(index=[0, 1], weight=1)

        # Place prediction text top left
        self._net_pred_txt_lbl.grid(row=0, column=0, sticky='ew')
    
        # Place prediciton number right of prediction text
        self._net_pred_num_lbl.grid(row=0, column=1, sticky='nsew')

        # Place confidence text below prediction info
        self._net_conf_txt_lbl.grid(row=1, column=0, sticky='ew')

        # Place prediction output right of prediction text
        self._net_conf_num_lbl.grid(row=1, column=1, sticky='nsew')

        # Place confidence graph below all the above, spanning both columns
        self._conf_figure_canvas.draw() # Acts like a canvas, must be drawn first
        self._conf_figure_canvas.get_tk_widget().grid(row=2, columnspan=2, sticky='nsew')
    
    # -----------------------
    # Buttons
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
        self._selected_net_acc_lbl.configure(text=f'Accuracy: {avg_acc}')
        self._selected_net_loss_lbl.configure(text=f'Loss: {avg_loss}')
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

        # Update the active network name label
        self._active_net_name_lbl.configure(text=f'Active Network: {net_name}')

        # Update the metrics labels with the new values
        self._active_net_acc_lbl.configure(text=f'Accuracy: {avg_acc}')
        self._active_net_loss_lbl.configure(text=f'Loss: {avg_loss}')
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


    def _on_u_tab_upload_click(self):

        # temp for easier testing
        if ScannerFrame.SKIP_IMG_SELECT:
            img_path = r"C:\Users\ikram\Documents\Schoolwork\compsci\test_img.png"
        else:
            img_path = ctk.filedialog.askopenfilename(title="Open Image File", 
                                          filetypes=[("Image files", "*.png *.jpg *.jpeg *.gif *.bmp *.ico")])

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
        usr_draw.show()
        return usr_draw


    def _set_formatted_img(self, img: Image.Image):
        '''
        Resize, conv to cp arr, and invert the given PIL image.
        Set the img and its cp arr as attributes.
        '''

        # Resize, greyscale
        img = img.resize(size=self.controller.DIGIT_IMG_SIZE).convert(mode='L')
        self._curr_img_pil = ImageOps.invert(img)

        # Convert to cp arr (0-1)
        self._curr_img_arr = cp.asarray(self._curr_img_arr) / 255

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

        # Move conf scores to CPU
        conf_scores_cpu = cp.asnumpy(conf_scores_gpu)

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

        self.controller.network.fwd(inputs)
        pred_digit, main_conf_score = self.controller.network.get_pred_digit_and_conf()


        # Update main prediction
        self._net_pred_num_lbl.config(text=f'{pred_digit}')

        # Update conf score and format to 1 d.p.
        self._net_conf_num_lbl.config(text=f'{main_conf_score:.1f}%')

        # Update figure with conf score from gpu and pred digit
        self._update_conf_figure(self.controller.network.conf_scores, pred_digit)