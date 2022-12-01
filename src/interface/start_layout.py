# initial start-up layout
from kivy.uix.screenmanager import Screen
from kivy.lang import Builder

from . import custom_button

Builder.load_file("src/interface/layouts/start_layout.kv")


class StartLayout(Screen):
    pass
