# file loading layout
from kivy.uix.screenmanager import Screen
from kivy.properties import ObjectProperty
from kivy.lang import Builder

from . import custom_button, error_popup

import os

Builder.load_file("src/interface/layouts/add_layout.kv")


class AddLayout(Screen):
    audio_file = ObjectProperty()
    csv_file = ObjectProperty()
    save_file = ObjectProperty()
    filechooser = ObjectProperty()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.text_input = self.audio_file

    def cancel(self):
        self.manager.transition.direction = "right"
        if len(self.manager.get_screen("MainLayout").children) > 0:
            self.manager.current = "MainLayout"
        else:
            self.manager.current = "StartLayout"

    def load(self):
        try:
            # check file paths are valid
            audio_file: str = self.audio_file.text
            csv_file: str = self.csv_file.text
            save_file: str = self.save_file.text
            use_csv: bool = self.ids["csv_checkbox"].active
            save_csv: bool = self.ids["save_checkbox"].active
            _, file_ext = os.path.splitext(audio_file)
            if file_ext.lower() != ".wav":
                raise ValueError("audio file is not a .wav")
            if not os.path.exists(audio_file):
                raise ValueError("audio file doesn't exist")
            if use_csv:
                _, csv_file_ext = os.path.splitext(csv_file)
                if csv_file_ext.lower() != ".csv":
                    raise ValueError("csv file is not a .csv")
                if not os.path.exists(csv_file):
                    raise ValueError("csv file doesn't exist")
            if save_csv:
                save_path, save_file_ext = os.path.splitext(save_file)
                if save_file_ext == "":
                    save_file_ext = ".csv"
                if save_file_ext != ".csv":
                    raise ValueError("save file is not a .csv")
                save_file = save_path + save_file_ext

            self.manager.transition.direction = "left"
            self.manager.current = "MainLayout"
            self.manager.ids["main_layout"].addStream(
                audio_file, csv_file, save_file, use_csv, save_csv
            )
        except ValueError as exc:
            error = error_popup.ErrorPopup(str(exc))
            error.open()
