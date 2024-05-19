from detector_script import calculate_frequencies
from dynamic_test import *
import numpy as np
import soundfile as sf

wf_data, rate = sf.read(f"./samples/long_d0_s0.flac")
wf = [x for x in wf_data[:,1]]

# wf = np.ndarray(wf_data.shape[0], )
# bars = calculate_frequencies(wf, binc, lambda e: e >= threshold and e <= cap)

class FrequencyFunction(PlotterFunction):
    def __init__(self):
        self.binc = 30
        self.data = wf[:len(wf)//3]
        self.threshold = .3
        self.cap = .9

    def domain(self) -> list:
        m = min(self.data)
        bin_width = (max(self.data) - min(self.data)) / int(self.binc)
        # bars = calculate_frequencies(self.data, int(self.binc), lambda e: e >= self.threshold and e <= self.cap)
        return [ ((m + i * bin_width) + (m + (i+1) * bin_width)) / 2.0 for i in range(int(self.binc)) ]

    def series(self) -> list:
        bars = calculate_frequencies(self.data, int(self.binc), lambda e: e >= self.threshold and e <= self.cap)
        return [b.freq for b in bars]

    def get_variables(self) -> list:
        return [
            ('threshold', 0.0, 1.0),
            ('cap', 0.0, 1.0),
            ('binc', 1, 100)
        ]

if __name__ == '__main__':
    k = Plotter()
    k.plot(FrequencyFunction())