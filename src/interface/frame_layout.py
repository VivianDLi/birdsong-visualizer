# container to display an audio stream
from kivy.uix.boxlayout import BoxLayout
from kivy.properties import ObjectProperty, NumericProperty
from kivy.lang import Builder

Builder.load_file("src/interface/layouts/frame_layout.kv")

from . import spectrogram, custom_button
from src.tools.interfaces import IAudioStream, ICoordinator


class FrameLayout(BoxLayout):
    spectrogram = ObjectProperty()
    offset = NumericProperty(0)

    def __init__(
        self, stream: IAudioStream, coordinator: ICoordinator, **kwargs
    ):
        self.stream = stream
        self.coordinator = coordinator
        super().__init__(**kwargs)

    def play(self):
        self.stream.play(self.offset)

    def stop(self):
        self.stream.stop()
