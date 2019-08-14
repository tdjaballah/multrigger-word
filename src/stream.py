import pyaudio
import numpy as np
import time
import tensorflow as tf

from matplotlib import pyplot as plt
from matplotlib import mlab as mlab

from src.settings.general import FRAME_RATE, FIGURE_DIR, NFFT
from src.settings.trigger import TX, FX, N_CLASSES, TRIGGER_KERNEL_SIZE, TRIGGER_STRIDE
from src.models import trigger_model


class SWHear(object):
    """
    The SWHear class is made to provide access to continuously recorded
    (and mathematically processed) microphone data.
    """

    def __init__(self, model):
        """fire up the SWHear class."""
        print(" -- initializing SWHear")

        self.chunk = 4096           # number of data points to read at a time
        self.rate = FRAME_RATE      # time resolution of the recording device (Hz)

        self.model = model
        # for tape recording (continuous "tape" of recent audio)
        self.tapeLength = 5         #seconds
        self.tape = np.empty(self.rate*self.tapeLength)*np.nan

        self.p = pyaudio.PyAudio()  # start the PyAudio class
        self.stream = self.p.open(format=pyaudio.paInt16,
                                  channels=1,
                                  rate=self.rate, input=True,
                                  frames_per_buffer=self.chunk)

    # LOWEST LEVEL AUDIO ACCESS
    # pure access to microphone and stream operations
    # keep math, plotting, FFT, etc out of here.

    def stream_read(self):
        """return values for a single chunk"""
        data = np.fromstring(self.stream.read(self.chunk, exception_on_overflow = False), dtype=np.int16)

        return data

    def stream_stop(self):
        """close the stream but keep the PyAudio instance alive."""
        if 'stream' in locals():
            self.stream.stop_stream()
            self.stream.close()
        print(" -- stream CLOSED")

    def close(self):
        """gently detach from things."""
        self.stream_stop()
        self.p.terminate()

    ### TAPE METHODS
    # tape is like a circular magnetic ribbon of tape that's continously
    # recorded and recorded over in a loop. self.tape contains this data.
    # the newest data is always at the end. Don't modify data on the type,
    # but rather do math on it (like FFT) as you read from it.

    def tape_add(self):
        """add a single chunk to the tape."""
        self.tape[:-self.chunk] = self.tape[self.chunk:]
        self.tape[-self.chunk:] = self.stream_read()

    def tape_forever(self, refresh_period=1):
        t1 = 0

        try:
            while True:
                self.tape_add()
                if (time.time() - t1) > refresh_period:
                    t1 = time.time()
                    self.waveform_tape_plot()
                    self.plot_predictions()
                    print("plotting saving took %.02f ms" % ((time.time() - t1) * 1000))

        except Exception as e:
            print(" ~~ exception {}".format(e))
            return

    def waveform_tape_plot(self, filename="{}/waveform.png".format(FIGURE_DIR)):
        """plot what's in the tape."""
        plt.plot(np.arange(len(self.tape))/self.rate, self.tape)
        plt.axis([0, self.tapeLength, -2**16/2, 2**16/2])

        plt.savefig(filename, dpi=50)

        plt.close('all')

    def spectrogram_tape_plot(self):

        specgram, _, _ = mlab.specgram(self.tape, NFFT, 2, noverlap=int(NFFT / 2))

        return specgram

    def plot_predictions(self, filename="{}/predictions.png".format(FIGURE_DIR)):

        x = self.spectrogram_tape_plot()
        np.nan_to_num(x, copy=False)
        x = np.swapaxes(x, 0, 1)
        x = np.expand_dims(x, axis=0)
        y = self.model.predict(x)[0]
        plt.plot(y[:, 1])
        plt.savefig(filename, dpi=50)
        plt.close('all')


if __name__ == "__main__":

    latest = tf.train.latest_checkpoint("../logs/trigger/checkpoints")

    model = trigger_model(input_shape=(TX, FX),
                          n_classes=N_CLASSES,
                          kernel_size=TRIGGER_KERNEL_SIZE,
                          stride=TRIGGER_STRIDE)

    model.load_weights(latest)
    ear = SWHear(model)
    ear.tape_forever()
    ear.close()
    print("DONE")
