# container to display an audio stream
from kivy.uix.boxlayout import BoxLayout
from kivy.properties import ObjectProperty, NumericProperty, StringProperty
from kivy.lang import Builder
from kivy.clock import Clock

from multiprocessing.pool import Pool
import traceback

Builder.load_file("src/interface/layouts/frame_layout.kv")

from . import spectrogram, custom_button
from src.tools.interfaces import IAudioStream, ICoordinator


class FrameLayout(BoxLayout):
    spectrogram = ObjectProperty()
    r_index = StringProperty("SpDiv")
    g_index = StringProperty("ACI")
    b_index = StringProperty("HfVar")

    offset = NumericProperty(0)
    playback_time = ObjectProperty()

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

    def on_offset(self, *args):
        offset_segment = self.offset // self.stream.segment_duration
        self.spectrogram.setOffset(offset_segment)

    def on_processing(self, *args):
        if self.processing == 0:
            self.loading.color = (1, 1, 1, 1)
            self.loading.source = "src/interface/resources/checkmark.png"
        else:
            self.loading.color = (1, 1, 1, 1)
            self.loading.source = "src/interface/resources/loading.gif"

    def _loadIndicesCallback(self, result):
        self.processing -= 1
        self.coordinator.spectrogram = result
        available_indices = result.getIndices()
        if self.r_index not in available_indices:
            self.r_index = available_indices[0]
        if self.g_index not in available_indices:
            self.g_index = available_indices[0]
        if self.b_index not in available_indices:
            self.b_index = available_indices[0]
        self.spectrogram.setSpectrogram(self.coordinator.spectrogram)

    def _calculateIndicesCallback(self, results):
        self.processing -= 1
        i, result, save_csv, save_file = results
        if not hasattr(self.coordinator.spectrogram, "shape"):
            self.coordinator.spectrogram.shape = (
                self.stream.getNumberOfSegments(),
                len(list(result.values())[0]),
            )
        self.coordinator.spectrogram.addSegment(i, result)
        print("added segment %d" % (i))
        if save_csv:
            self.coordinator.saveIndices(save_file)
        available_indices = self.coordinator.spectrogram.getIndices()
        if self.r_index not in available_indices:
            self.r_index = available_indices[0]
        if self.g_index not in available_indices:
            self.g_index = available_indices[0]
        if self.b_index not in available_indices:
            self.b_index = available_indices[0]
        self.spectrogram.setSpectrogram(self.coordinator.spectrogram)

    def calculateIndices(self, use_csv, csv_file, save_csv, save_file):
        if use_csv:
            self._pool.apply_async(
                _loadIndices,
                args=(self.coordinator, csv_file, save_csv, save_file),
                callback=self._loadIndicesCallback,
                error_callback=lambda exc: traceback.print_exc(),
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
                    error_callback=lambda exc: traceback.print_exc(),
                )
                self.processing += 1

    def setOffset(self, offset: int):
        true_offset = offset - self.stream.time_limits[0]
        return max(
            min(
                true_offset,
                int(self.stream.getDuration()),
            ),
            0,
        )

    def setPlaybackTime(self, time: int):
        self.playback_time.text = str(time)

    def play(self):
        playback_thread = self.stream.play(self.offset)
        self._playback_clock = Clock.schedule_interval(
            lambda _: self.setPlaybackTime(
                round(playback_thread.playback_time)
            ),
            1 / 2.0,
        )

    def stop(self):
        self.stream.stop()
        if hasattr(self, "_playback_clock"):
            self._playback_clock.cancel()


def _loadIndices(coordinator, csv_file, save_csv, save_file):
    coordinator.loadIndices(csv_file)
    if save_csv:
        coordinator.saveIndices(save_file)
    return coordinator.getSpectrogram()


def _calculateSegment(
    coordinator, i, r_index, g_index, b_index, save_csv, save_file
):
    result = coordinator.calculateSegment(i, r_index, g_index, b_index)
    return (i, result, save_csv, save_file)
