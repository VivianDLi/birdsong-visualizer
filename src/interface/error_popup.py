from kivy.uix.popup import Popup
from kivy.lang import Builder

Builder.load_file("src/interface/layouts/error_popup.kv")


class ErrorPopup(Popup):
    def __init__(self, error_text: str, **kwargs):
        self.error_text = error_text
        super().__init__(**kwargs)
