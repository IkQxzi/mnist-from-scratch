import time

from app_ui.controller import App
from app_ui.frames.scanner import ScannerFrame
from app_ui.frames.menu import MenuFrame

app = App()
app.mainloop()
app.show_frame(MenuFrame)

