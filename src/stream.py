import pyaudio
import time
import pylab
import numpy as np

from src.settings.general import FRAME_RATE, FIGURE_DIR


class SWHear(object):
    """
    The SWHear class is made to provide access to continuously recorded
    (and mathematically processed) microphone data.
    """

    def __init__(self):
        """fire up the SWHear class."""
        print(" -- initializing SWHear")

        self.chunk = 4096           # number of data points to read at a time
        self.rate = FRAME_RATE      # time resolution of the recording device (Hz)

        # for tape recording (continuous "tape" of recent audio)
        self.tapeLength = 2         #seconds
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
        data = np.fromstring(self.stream.read(self.chunk), dtype=np.int16)

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

    def tape_forever(self, plot_period=.25):
        init = 0
        try:
            while True:
                self.tape_add()
                if (time.time()-init) > plot_period:
                    init = time.time()
                    self.tape_plot()
        except:
            print(" ~~ exception (keyboard?)")
            return

    def tape_plot(self, filename="{}/figure.png".format(FIGURE_DIR)):
        """plot what's in the tape."""
        pylab.plot(np.arange(len(self.tape))/self.rate,self.tape)
        pylab.axis([0, self.tapeLength, -2**16/2, 2**16/2])

        if filename:
            t1 = time.time()
            pylab.savefig(filename, dpi=50)
            print("plotting saving took %.02f ms" % ((time.time()-t1)*1000))

        else:
            pylab.show()

        pylab.close('all')


if __name__ == "__main__":
    ear = SWHear()
    ear.tape_forever()
    ear.close()
    print("DONE")
