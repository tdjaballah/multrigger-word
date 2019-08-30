import pyaudio
import numpy as np
import time
import tensorflow as tf
import pydub
import seaborn as sns

from matplotlib import pyplot as plt

from src.utils.misc_utils import transform_labels
from src.utils.tf_helper import _soft_f1_macro, f1_scores_1, f1_scores_2, f1_scores_3
from src.settings.general import *

SILENCE_THRESHOLD = 1000


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
        self.tapeLength = int(BACKGROUND_DURATION_MS / 1000)        #seconds
        self.tape = np.zeros(self.rate*self.tapeLength, dtype=np.float32)

        self.p = pyaudio.PyAudio()  # start the PyAudio class
        self.stream = self.p.open(format=pyaudio.paFloat32,
                                  channels=1,
                                  rate=self.rate, input=True,
                                  frames_per_buffer=self.chunk)

    # LOWEST LEVEL AUDIO ACCESS
    # pure access to microphone and stream operations
    # keep math, plotting, FFT, etc out of here.

    def stream_read(self):
        """return values for a single chunk"""
        data = np.fromstring(self.stream.read(self.chunk, exception_on_overflow = False), dtype=np.float32)

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

    def tape_forever(self, refresh_period=0.2):
        t1 = 0

        try:
            while True:
                self.tape_add()
                if (time.time() - t1) > refresh_period:
                    t1 = time.time()
                    self.waveform_tape_plot()
                    spectrogram = self.make_spectrogram(export=True)
                    self.make_prediction(spectrogram, LABEL_MAP_DICT)
                    print("plotting saving took %.02f ms" % ((time.time() - t1) * 1000))

        except Exception as e:
            print(" ~~ exception {}".format(e))
            return

    def waveform_tape_plot(self, filename="{}/waveform.png".format(FIGURE_DIR)):
        """plot what's in the tape."""
        plt.plot(np.arange(len(self.tape))/self.rate, self.tape)
        plt.axis([0, self.tapeLength, -1, 1])

        plt.savefig(filename, dpi=50)

        plt.close('all')

    def make_spectrogram(self, export=False, filename="{}/spectrogram.png".format(FIGURE_DIR)):

        x = self.tape

        audio = pydub.AudioSegment(x.tobytes(), frame_rate=FRAME_RATE, channels=1, sample_width=x.dtype.itemsize)
        waveform = np.array(audio.get_array_of_samples(), dtype=np.float32)

        signals = tf.reshape(waveform, [1, -1])
        stfts = tf.contrib.signal.stft(signals, frame_length=FFT_FRAME_LENGTH, frame_step=FFT_FRAME_STEP,
                                       fft_length=FFT_LENGTH)
        magnitude_spectrograms = tf.abs(stfts)

        num_spectrogram_bins = magnitude_spectrograms.shape[-1].value

        linear_to_mel_weight_matrix = tf.contrib.signal.linear_to_mel_weight_matrix(NUM_MEL_BINS, num_spectrogram_bins,
                                                                                    FRAME_RATE, LOWER_EDGE_HERTZ,
                                                                                    UPPER_EDGE_HERTZ)

        mel_spectrograms = tf.tensordot(magnitude_spectrograms, linear_to_mel_weight_matrix, 1)
        log_mel_spectrograms = tf.log(mel_spectrograms + tf.keras.backend.epsilon())

        if export:
            sns_plot = sns.heatmap(np.swapaxes(log_mel_spectrograms.numpy()[0], 0, 1))
            sns_plot.get_figure().savefig(filename)
            plt.close('all')

        return log_mel_spectrograms

    def make_prediction(self, spectrogram, map_dict, export=True, filename="{}/predictions.png".format(FIGURE_DIR)):

        predictions = np.squeeze(model.predict(spectrogram))

        if export:
            predictions_df = transform_labels(predictions, map_dict)
            sns_plot = sns.relplot(x='x', y='y', row='label', data=predictions_df, kind='line', height=2, aspect=2)
            sns_plot.savefig(filename)
            plt.close('all')

        return predictions


if __name__ == "__main__":

    tf.enable_eager_execution()

    latest = tf.train.latest_checkpoint("../logs/checkpoints")

    model = tf.keras.models.load_model('{}/model.h5'.format(LOG_DIR), custom_objects={'_soft_f1_macro': _soft_f1_macro,
                                                                                      'f1_scores_1': f1_scores_1,
                                                                                      'f1_scores_2': f1_scores_2,
                                                                                      'f1_scores_3': f1_scores_3})

    model.load_weights(latest)

    print(model.summary())
    ear = SWHear(model)
    ear.tape_forever()
    ear.close()
    print("DONE")
