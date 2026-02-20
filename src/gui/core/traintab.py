import customtkinter as ctk


class TrainingTab(ctk.CTkTabview):
    
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)

        self.add()

