# switch between different layouts
from kivy.uix.screenmanager import ScreenManager
from kivy.properties import ObjectProperty
from kivy.lang import Builder

Builder.load_file("src/interface/layouts/layout_manager.kv")

from . import start_layout, add_layout, main_layout


class LayoutManager(ScreenManager):
    pass
