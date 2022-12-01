# file loading layout
from kivy.uix.screenmanager import Screen
from kivy.properties import ObjectProperty
from kivy.lang import Builder

from . import custom_button, error_popup
from src.tools.loader import load_audio
from src.analysis.coordinator import AnalysisCoordinator

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
        self.manager.current = "StartLayout"

    def load(self):
        try:
            stream = load_audio(self.audio_file.text)
            coordinator = AnalysisCoordinator(stream, ["Ht", "Hf", "ACI"])
            if self.ids["csv_checkbox"].active:
                coordinator.loadIndices(self.csv_file.text)
            else:
                coordinator.calculateIndices()
            if self.ids["save_checkbox"].active:
                coordinator.saveIndices(self.save_file.text)
            self.manager.ids["main_layout"].add_stream(stream, coordinator)
            self.manager.transition.direction = "left"
            self.manager.current = "MainLayout"
        except ValueError as exc:
            error = error_popup.ErrorPopup(str(exc))
            error.open()
