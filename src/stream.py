import logging
import matplotlib.mlab as mlab
import numpy as np
import pyaudio
import tensorflow as tf
import time

from queue import Queue

from src.make_model import trigger_model
from src.settings import *


def load_model(weights_dir):
    """
    Load our seq_model with the latest checkpoint
    :param weights_dir: directory where we have our checkpoints from our training
    :return: our sequence model with weights
    """
    latest = tf.train.latest_checkpoint(str(weights_dir))
    model = trigger_model(input_shape=(TX, FX),
                          n_classes=N_WORDS,
                          kernel_size=KERNEL_SIZE,
                          stride=STRIDE)

    model.load_weights(latest)

    return model


def predict(model, x):
    """
    Function to predict the location of one of the trigger word.
    :param x: spectrum of shape (TX, FX)
    :return: numpy array to shape (number of output time steps)
    """

    x = np.expand_dims(x, axis=0)
    predictions = model.predict(x)

    return predictions


def get_audio_input_stream(callback):
    stream = pyaudio.PyAudio().open(
        format=pyaudio.paInt16,
        channels=1,
        rate=FS,
        input=True,
        frames_per_buffer=CHUNK_SAMPLES,
        input_device_index=0,
        stream_callback=callback)
    return stream


def detect_triggerword_spectrum(model, x):
    """
    Function to predict the location of the trigger word.

    :param model: neural network that is use for inference
    :param x: spectrum of shape (TX, FX)
    :return: predictions -- numpy ndarray to shape (number of output time steps per num_classes)
    """

    x = np.expand_dims(x, axis=0)
    predictions = model.predict(x)
    return predictions.reshape(-1)


def has_new_triggerword(predictions, chunk_duration, feed_duration, threshold=0.5):
    """
    Function to detect new trigger word in the latest chunk of input audio.
    It is looking for the rising edge of the predictions data belongs to the
    last/latest chunk.

    :param predictions:  predicted labels from model
    :param chunk_duration: time in second of a chunk
    :param feed_duration: time in second of the input to model
    :param threshold: threshold for probability above a certain to be considered positive
    :return: True if new trigger word detected in the latest chunk
    """

    predictions = predictions > threshold
    chunk_predictions_samples = int(len(predictions) * chunk_duration / feed_duration)
    chunk_predictions = predictions[-chunk_predictions_samples:]
    level = chunk_predictions[0]
    for pred in chunk_predictions:
        if pred > level:
            return True
        else:
            level = pred
    return False


def get_spectrogram(data):
    """
    Function to compute a spectrogram.
    :param data: one channel / dual channel audio data as numpy array
    :return: spectrogram, 2-D array, columns are the periodograms of successive segments.
    """

    nfft = 200  # Length of each window segment
    fs = 8000  # Sampling frequencies
    noverlap = 120  # Overlap between windows
    nchannels = data.ndim
    if nchannels == 1:
        pxx, _, _ = mlab.specgram(data, nfft, fs, noverlap=noverlap)
    elif nchannels == 2:
        pxx, _, _ = mlab.specgram(data[:, 0], nfft, fs, noverlap=noverlap)
    return pxx


def main():

    model = load_model(TRIGGER_CHECKPOINT_DIR)
    print(type(model))

    # Queue to communicate between the audio callback and main thread
    q = Queue()

    run = True

    silence_threshold = 100

    # Run the demo for a timeout seconds
    timeout = time.time() + 0.5 * 60  # 0.5 minutes from now

    # Data buffer for the input wavform
    data_stream = np.zeros(FEED_SAMPLES, dtype='int16')

    def callback(in_data, frame_count, time_info, status):

        global run, data_stream, timeout, silence_threshold

        if time.time() > timeout:
            run = False

        data0 = np.frombuffer(in_data, dtype='int16')

        if np.abs(data0).mean() < silence_threshold:
            print('-')
            return in_data, pyaudio.paContinue
        else:
            print('.')

        data_stream = np.append(data_stream, data0)

        if len(data_stream) > FEED_SAMPLES:
            data = data_stream[-FEED_SAMPLES:]
            # Process data async by sending a queue.
            q.put(data)
        return in_data, pyaudio.paContinue

    stream = get_audio_input_stream(callback)
    stream.start_stream()
    print("start streaming")

    try:
        while run:
            data_stream = q.get()
            spectrum = get_spectrogram(data_stream)
            preds = detect_triggerword_spectrum(model, spectrum)
            new_trigger = has_new_triggerword(preds, CHUNK_DURATION, FEED_DURATION)
            if new_trigger:
                print('1')
    except (KeyboardInterrupt, SystemExit):
        stream.stop_stream()
        stream.close()
        timeout = time.time()
        run = False

    stream.stop_stream()
    stream.close()


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
