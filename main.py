import time

from app_ui.controller import App
from app_ui.frames.scanner import ScannerFrame

app = App()
app.mainloop()
app.show_frame(ScannerFrame)

