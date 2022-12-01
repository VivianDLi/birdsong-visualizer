# base layout after loading a file
from kivy.uix.screenmanager import Screen
from kivy.properties import ObjectProperty
from kivy.lang import Builder

Builder.load_file("src/interface/layouts/main_layout.kv")

from . import frame_layout
from src.tools.interfaces import IAudioStream, ICoordinator


class MainLayout(Screen):
    layout = ObjectProperty()

    def add_stream(self, stream: IAudioStream, coordinator: ICoordinator):
        frame = frame_layout.FrameLayout(stream, coordinator)
        self.layout.add_widget(frame)
