# toolbar for adding new files or changing settings
from kivy.uix.stacklayout import StackLayout
from kivy.properties import ObjectProperty
from kivy.lang import Builder

from . import custom_button

Builder.load_file("src/interface/layouts/toolbar.kv")


class Toolbar(StackLayout):
    manager = ObjectProperty()
