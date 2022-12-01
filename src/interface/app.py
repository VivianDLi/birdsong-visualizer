# GUI application using Kivy

import kivy

kivy.require("2.1.0")

from kivy.app import App
from kivy.properties import ObjectProperty
from kivy.lang import Builder

from . import layout_manager, toolbar


class MyApp(App):
    def build(self):
        return Builder.load_file("src/interface/layouts/app.kv")
