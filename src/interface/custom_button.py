# custom button template
from kivy.uix.button import Button
from kivy.lang import Builder

Builder.load_file("src/interface/layouts/custom_button.kv")


class CustomButton(Button):
    pass
