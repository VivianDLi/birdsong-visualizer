# widget to represent a spectrogram drawing
from kivy.uix.widget import Widget
from kivy.properties import ObjectProperty, NumericProperty, ListProperty
from kivy.lang import Builder
from kivy.graphics import Color, Rectangle

Builder.load_file("src/interface/layouts/spectrogram.kv")


class Spectrogram(Widget):
    spectrogram = ObjectProperty()
    indices = ListProperty()
    shape = ListProperty()
    offset = NumericProperty()

    def update_canvas(self, *args):
        gap = (self.size[0] / self.shape[0], self.size[1] / self.shape[1])
        self.canvas.clear()  # type:ignore
        with self.canvas:  # type:ignore
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    Color(
                        self.spectrogram.getColorResult(self.indices[0])[i][j],
                        self.spectrogram.getColorResult(self.indices[1])[i][j],
                        self.spectrogram.getColorResult(self.indices[2])[i][j],
                    )
                    Rectangle(
                        pos=(
                            self.pos[0] + i * gap[0],
                            self.pos[1] + j * gap[1],
                        ),
                        size=gap,
                    )
