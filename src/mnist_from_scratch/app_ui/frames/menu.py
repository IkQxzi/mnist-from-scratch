from typing import TYPE_CHECKING

import customtkinter as ctk

from mnist_from_scratch.app_ui.frames.scanner import ScannerFrame

if TYPE_CHECKING:
    from mnist_from_scratch.app_ui.controller import App


class MenuFrame(ctk.CTkFrame):
    controller: "App"

    def __init__(self, container, controller: "App"):

        # Put frame inside of container
        super().__init__(container)

        self.controller = controller

        # Create widgets
        self._create_widgets()
        self._layout_widgets()

    # -----------------------
    # Creation
    # -----------------------

    def _create_widgets(self):

        self._scanner_btn = ctk.CTkButton(
            self,
            text="Open Scanner",
            font=self.controller.BIGGEST_BTN_FONT,
            height=self.controller.BTN_HEIGHT,
            command=self._on_scanner_click,
        )

        # Red to show destructive
        self._quit_btn = ctk.CTkButton(
            self,
            text="Quit Application",
            font=self.controller.BIGGEST_BTN_FONT,
            fg_color="#ef4444",
            hover_color="#b91c1c",
            command=self._on_quit_click,
        )

    # -----------------------
    # Layout
    # -----------------------

    def _layout_widgets(self):

        # Configure grid
        self.grid_rowconfigure((0, 3), weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Place the btns
        self._scanner_btn.grid(row=1, column=0, pady=10)
        self._quit_btn.grid(row=2, column=0, pady=10)

    # -----------------------
    # Callback
    # -----------------------

    def _on_scanner_click(self):

        # Move to scanner frame
        self.controller.show_frame(ScannerFrame)

    def _on_quit_click(self):

        # Close app
        self.controller.destroy()
