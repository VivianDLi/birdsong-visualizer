# container to display an audio stream
from kivy.uix.boxlayout import BoxLayout
from kivy.properties import ObjectProperty, NumericProperty, StringProperty
from kivy.lang import Builder

from multiprocessing.pool import Pool
import traceback

Builder.load_file("src/interface/layouts/frame_layout.kv")

from . import spectrogram, custom_button
from src.tools.interfaces import IAudioStream, ICoordinator, ISpectrogram


class FrameLayout(BoxLayout):
    spectrogram = ObjectProperty()
    r_index = StringProperty("SpDiv")
    g_index = StringProperty("ACI")
    b_index = StringProperty("HfVar")

    offset = NumericProperty(0)

    processing = NumericProperty(0)
    loading = ObjectProperty()

    def __init__(
        self, stream: IAudioStream, coordinator: ICoordinator, **kwargs
    ):
        self.stream = stream
        self.coordinator = coordinator
        self._pool = Pool(processes=6)
        super().__init__(**kwargs)

    def on_r_index(self, *args):
        self.spectrogram.indices[0] = self.r_index

    def on_g_index(self, *args):
        self.spectrogram.indices[1] = self.g_index

    def on_b_index(self, *args):
        self.spectrogram.indices[2] = self.b_index

    def on_processing(self, *args):
        if self.processing == 0:
            self.loading.color = (1, 1, 1, 0)
        else:
            self.loading.color = (1, 1, 1, 1)

    def _calculateIndicesCallback(self, result):
        self.processing -= 1
        self.coordinator.spectrogram = result
        available_indices = result.getIndices()
        if self.r_index not in available_indices:
            self.r_index = available_indices[0]
        if self.g_index not in available_indices:
            self.g_index = available_indices[0]
        if self.b_index not in available_indices:
            self.b_index = available_indices[0]
        self.spectrogram.setSpectrogram(result)

    def calculateIndices(self, use_csv, csv_file, save_csv, save_file):
        if use_csv:
            self._pool.apply_async(
                _loadIndices,
                args=(self.coordinator, csv_file, save_csv, save_file),
                callback=self._calculateIndicesCallback,
                error_callback=lambda exc: print(
                    "Loading .csv at %s generated an exception: %s"
                    % (csv_file, exc)
                ),
            )
            self.processing += 1
        else:
            for i in range(self.stream.getNumberOfSegments()):
                self._pool.apply_async(
                    _calculateSegment,
                    args=(
                        self.coordinator,
                        i,
                        self.r_index,
                        self.g_index,
                        self.b_index,
                        save_csv,
                        save_file,
                    ),
                    callback=self._calculateIndicesCallback,
                    error_callback=lambda exc: print(
                        "Segment starting at %r generated an exception: %s"
                        % (self.stream.segmentToTimestamp(i), exc)
                    ),
                )
                self.processing += 1

    def play(self):
        self.stream.play(self.offset)

    def stop(self):
        self.stream.stop()


def _loadIndices(coordinator, csv_file, save_csv, save_file):
    coordinator.loadIndices(csv_file)
    if save_csv:
        coordinator.saveIndices(save_file)
    return coordinator.getSpectrogram()


def _calculateSegment(
    coordinator, i, r_index, g_index, b_index, save_csv, save_file
):
    coordinator.calculateSegment(i, r_index, g_index, b_index)
    if save_csv:
        coordinator.saveIndices(save_file)
    return coordinator.getSpectrogram()
