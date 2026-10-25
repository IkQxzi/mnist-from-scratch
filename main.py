import time

from mnist_from_scratch.app_ui.controller import App
from mnist_from_scratch.app_ui.frames.scanner import ScannerFrame
from mnist_from_scratch.app_ui.frames.menu import MenuFrame

app = App()
app.mainloop()
app.show_frame(MenuFrame)

