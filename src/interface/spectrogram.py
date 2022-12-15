# widget to represent a spectrogram drawing
from kivy.clock import mainthread
from kivy.uix.anchorlayout import AnchorLayout
from kivy.properties import ObjectProperty, ListProperty
from kivy.lang import Builder
from kivy.uix.image import Image
from kivy.graphics.texture import Texture

import numpy as np

Builder.load_file("src/interface/layouts/spectrogram.kv")

from src.tools.interfaces import ISpectrogram


class Spectrogram(AnchorLayout):
    spectrogram = ObjectProperty()
    indices = ListProperty(["SpDiv", "ACI", "HfVar"])
    scrollview = ObjectProperty()

    @mainthread
    def setSpectrogram(self, spectrogram: ISpectrogram):
        self.spectrogram = spectrogram

    @mainthread
    def setImage(self, image: Image):
        for child in self.scrollview.children:
            self.scrollview.remove_widget(child)
        self.scrollview.add_widget(image)

    def on_spectrogram(self, *args):
        if self.spectrogram is not None:
            # get colors and convert to ubyte
            r = self.spectrogram.getColorResult(self.indices[0]) * 255.999
            g = self.spectrogram.getColorResult(self.indices[1]) * 255.999
            b = self.spectrogram.getColorResult(self.indices[2]) * 255.999
            img_data = np.dstack((r, g, b)).astype(np.uint8)
            # swap axes - numpy shape needs to be the transpose of the texture
            img_data = np.swapaxes(img_data, 0, 1)
            index, freq, _ = img_data.shape
            texture = Texture.create(size=(index, freq), colorfmt="rgb")
            texture.blit_buffer(
                img_data.flatten(), colorfmt="rgb", bufferfmt="ubyte"
            )
            image = Image(
                size=(index, self.height),
                texture=texture,
                allow_stretch=True,
                keep_ratio=False,
            )
            image.size_hint_x = None
            self.setImage(image)
        else:
            image = Image(source="src/interface/resources/loading.gif")
            image.size_hint_x = None
            self.setImage(image)
