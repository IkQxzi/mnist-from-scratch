import os
import json
from typing import Type

import numpy as np
import customtkinter as ctk

from mnist_from_scratch.neural import NetworkDense

# from gui.frames.menu import MenuFrame
# from gui.frames.trainer import TrainerFrame
from mnist_from_scratch.app_ui.frames.scanner import ScannerFrame
from mnist_from_scratch.app_ui.frames.menu import MenuFrame


class App(ctk.CTk):
    # Constants
    WIN_TITLE = "Handwritten Digit Recogniser Thingy"
    RESOLUTION = "1920x1080"
    UPDATE_INTERVAL = 1000  # ms
    MODEL_DIR = (
        r"C:\Users\ikram\Documents\Schoolwork\compsci\NEA 2026\src\training\models"
    )
    DIGIT_IMG_SIZE = (28, 28)

    # Button sizes
    BTN_HEIGHT = 50
    SMALL_BTN_HEIGHT = 20

    # Fonts
    SMALL_BTN_FONT = ("Arial", 14, "bold")
    BTN_FONT = ("Arial", 25, "bold")
    BIG_BTN_FONT = ("Arial", 35, "bold")
    BIGGEST_BTN_FONT = ("Arial", 45, "bold")

    # Styling constants
    PRIMARY_BTN_COLOUR = "#1F6AA5"
    PRIMARY_LBL_COLOUR = "#304674"
    PRIMARY_BG_COLOUR = "#2b2b2b"
    PRIMARY_GREEN_COLOUR = "#22c55e"
    SECONDARY_BLUE_COLOUR = "#3b82f6"
    LBL_CORNER_RADIUS = 10

    # To be used by the frames
    container: ctk.CTkFrame
    loaded_frames: dict[type[ctk.CTkFrame], ctk.CTkFrame]
    network: NetworkDense
    active_net_info: dict[str, str | int]
    net_infos: dict[str, list[str]]

    def __init__(self):
        """
        Docstring for __init__

        :param frame_list: List of
        :type frame_list: list[ctk.CTkFrame]
        """

        super().__init__()
        print("Updating net infos...")

        self._setup_window()

        # Update network infos for frames to use
        # Note that this must be done before loading the container
        self.update_net_infos()

        # Contains frames
        self._load_container()

    def show_frame(self, frame_class: type[ctk.CTkFrame]):
        """Bring the specified frame to the front of container, displaying it."""

        frame = self.loaded_frames[frame_class]
        frame.tkraise()  # Bring to front/display

    def set_network(self, net_name: str) -> dict:
        """
        Set the active network to the one described by the specified network name,
        and return the metadata about that network and its training.
        """

        # To keep linter happy
        net_fname = ""

        # Get the file name of network from just the name itself
        # Raise error if file not found
        net_index = self.net_infos["net_names"].index(net_name)
        net_fname = self.net_infos["net_fnames"][net_index]

        # Load network container and extract metadata
        raw_net_cont = np.load(file=rf"{App.MODEL_DIR}\{net_fname}")
        raw_metadata_json_str = str(raw_net_cont["metadata"])
        metadata = json.loads(raw_metadata_json_str)
        training_metadata = metadata["training"]

        # Change active network
        self.network = NetworkDense.load_pretrained(raw_net_cont, metadata["layers"])

        # Save active network info as controller attributes so all frames can access it
        # Mirrors structure in train/train_networks.py
        self.active_net_info = {
            "f_name": net_fname,
            "name": net_name,
            "avg_acc": training_metadata["avg_acc"],
            "avg_loss": training_metadata["avg_loss"],
            "size": len(metadata["layers"]),
        }

        # Return metadata
        return metadata

    def update_net_infos(self) -> None:
        """Update the dict of network infos in self."""

        self.net_infos = self._get_pretrained_networks()

    def _get_pretrained_networks(self) -> dict[str, list[str]]:
        """Get dict of extensionless file names, net names, net accuracies, net losses,
        sizes, timestamps of trained networks."""

        net_infos = {
            # Get file names at cls-specified directory
            "net_fnames": os.listdir(path=App.MODEL_DIR),
            "net_names": [],
            "net_accs": [],
            "net_losses": [],
            "net_sizes": [],
            "timestamps": [],
        }

        for f_name in net_infos["net_fnames"]:
            print(f"FILE NAME: {f_name}")

            # Separator has to mirror what is in train/train_networks.py
            # Underscores are fine because ISO8601 uses T as separator instead of _
            f_name_split = f_name.split("_")

            # Update net infos dict from parts of the filename
            net_infos["net_names"].append(f_name_split[0])
            net_infos["net_accs"].append(f_name_split[1])
            net_infos["net_sizes"].append(f_name_split[2])
            net_infos["net_losses"].append(f_name_split[3])
            net_infos["timestamps"].append(f_name_split[4])

        return net_infos

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
            fill="both",  # To make container fill both x & y directions in the window by default
            expand=True,  # To make container expand to fill the window if resized
        )

        # Make the grid inside the container (0, 0) expand to fill the container
        # By giving it 100% priority over free space (weight is distributed between grid items)
        self.container.grid_rowconfigure(index=0, weight=1)
        self.container.grid_columnconfigure(index=0, weight=1)

        self.loaded_frames = dict()

        # Initialise frames and put into self.loaded_frames

        for F in (ScannerFrame, MenuFrame):
            frame = F(container=self.container, controller=self)
            self.loaded_frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        for key, val in self.loaded_frames.items():
            print(key, val)
            print(type(key), type(val))
