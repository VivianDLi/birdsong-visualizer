# base layout after loading a file
from kivy.uix.screenmanager import Screen
from kivy.properties import ObjectProperty
from kivy.lang import Builder

Builder.load_file("src/interface/layouts/main_layout.kv")

from . import frame_layout
from src.tools.loader import load_audio
from src.analysis.coordinator import AnalysisCoordinator


class MainLayout(Screen):
    layout = ObjectProperty()

    def addStream(
        self,
        audio_file: str,
        csv_file: str,
        save_file: str,
        use_csv: bool,
        save_csv: bool,
    ):
        stream = load_audio(audio_file)
        coordinator = AnalysisCoordinator(stream)
        frame = frame_layout.FrameLayout(stream, coordinator)
        frame.calculateIndices(use_csv, csv_file, save_csv, save_file)
        self.layout.add_widget(frame)

    def on_layout(self, *args):
        for frame in self.layout.children:
            frame.spectrogram.on_spectrogram(*args)
