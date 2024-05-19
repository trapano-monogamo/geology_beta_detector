""" ~ threshold ~
with some off-by-one double checking and a way to move the window correctly, something
like this could work:
 + some_offset_like_.05
    thresholds = [
            max(median_filter( [e for e in wf[x0-window : x0+window+1]] ))
        for e in wf ]

but this would require a different filter() function from the built-in one, so that the filter function
can access the correct threshold or calculate it on the spot using the data from the signal.
"""


from os import walk
import math
# import threading

import numpy as np
from scipy.signal import find_peaks
import soundfile as sf

import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('Agg')



class EnergyClass:
    e_mix: float
    e_max: float
    freq: int

    def __init__(self, e_min: float, e_max: float, freq: int = 0):
        self.e_min = e_min
        self.e_max = e_max
        self.freq = freq

def calculate_frequencies(data: list, n_bins: int, _filter = lambda _: True) -> list:
    m = min(data)
    bin_width = (max(data) - min(data)) / n_bins
    histogram = [ EnergyClass(m + i * bin_width, m + (i+1) * bin_width) for i in range(n_bins) ]
    filtered = filter(_filter, data)
    for value in filtered:
        for bin in histogram:
            if value >= bin.e_min and value < bin.e_max:
                bin.freq += 1
    return histogram



class Config:
    threshold: float
    cap: float
    binc: int
    sample_filename: str
    distance: float
    shield: float

    def __init__(self, threshold,cap,binc,sample_filename,distance,shield):
        self.threshold = threshold
        self.cap = cap
        self.binc = binc
        self.sample_filename = sample_filename
        self.distance = distance
        self.shield = shield



output_dir = "./results"
samples_dir = "./samples/measurements"



def process_data(config: Config):
    data, rate = sf.read(f"{samples_dir}/{config.sample_filename}.flac")
    print(f"reading sample: {config.sample_filename}\tshape: {data.shape}, rate: {rate}")

    wf = [x for x in data[:,1]]
    # abs_wf = [abs(x) for x in data[:,1]]
    peaks, _ = find_peaks(wf, [config.threshold, config.cap])
    # bars1 = calculate_frequencies(wf, binc, lambda e: e >= config.threshold and e <= config.cap)
    bars = calculate_frequencies([wf[i] for i in peaks], binc, lambda e: e >= config.threshold and e <= config.cap)
    # filtered_wf = [wf[i] if i in peaks else 0 for i in range(len(wf))]

    fig, axs = plt.subplots(2, constrained_layout=True)

    axs[0].set_title("Detected signal")
    axs[0].set_xlabel(f"Samples [rate: {rate}/s]")
    axs[0].set_ylabel("Normalized intensity")
    axs[0].plot(wf, c='g', label="signal")
    axs[0].axhline(config.threshold, ls='--', c='tab:orange', label="threshold")
    axs[0].axhline(config.cap, ls='--', c='tab:red', label="cap")
    axs[0].scatter([idx for idx in peaks], [wf[idx] for idx in peaks])
    axs[0].legend()

    # axs[1].set_title("Energies spectrum")
    # axs[1].set_xlabel("Normalized classes")
    # axs[1].set_ylabel("Frequency")
    # axs[1].plot([(b.e_min + b.e_max)/2.0 for b in bars], [b.freq for b in bars], label="frequencies")
    # axs[1].legend()

    # axs[0].plot(list(range(0,len(filtered_wf))), filtered_wf, c='tab:green')
    # axs[0].axhline(config.threshold, ls='--', c='tab:orange', label="threshold")
    # axs[0].axhline(config.cap, ls='--', c='tab:red', label="cap")
    # axs[1].plot([(b.e_min + b.e_max)/2.0 for b in bars1], [b.freq for b in bars1], label="frequencies")
    axs[1].plot([(b.e_min + b.e_max)/2.0 for b in bars], [b.freq for b in bars], label="frequencies")

    plt.savefig(f"{output_dir}/{config.sample_filename}.png")
    plt.close(fig) # saves memory usage

    f = open(f"{output_dir}/{config.sample_filename}.dat", 'w')
    f.write(f"class_width: {bars[0].e_max - bars[0].e_min}\n")
    f.write(f"source_distance: {config.distance}\n")
    f.write(f"shield: {config.shield}\n")
    for i in range(len(bars)):
        f.write(f"{(bars[i].e_min + bars[i].e_max) / 2.0},{bars[i].freq}\n")
    f.close()



def calculate_energy_absorption(sample_filename: str, results_filename: str):
    sample_file = open(f"{output_dir}/{sample_filename}.dat", 'r')
    results_file = open(f"{output_dir}/{results_filename}.dat", 'a')
    
    raw_data = sample_file.read().split('\n')
    (width, dist, shield) = [float(x.split(": ")[1]) for x in raw_data[:3]]
    print(f"analyzing sample: {sample_filename}\twidth: {width}, dist: {dist}, shiled: {shield}")

    bars = []
    for row in raw_data[3:]:
        row = row.split(',')
        if (len(row)) == 2:
            (center, frequency) = [float(x) for x in row]
            bars.append(EnergyClass(center - width/2.0, center + width/2.0, frequency))

    energy_absorbed = 0
    for bar in bars:
        energy_absorbed += width * ((bar.e_min + bar.e_max) / 2.0) * bar.freq
    
    results_file.write(f"{sample_filename},{dist},{energy_absorbed}\n")

    sample_file.close()
    results_file.close()



def plot_results(results_filename: str):
    f = open(f"{output_dir}/{results_filename}.dat", 'r')
    data = f.read()
    dists = []
    energies = []
    # errors = []
    for row in data.split('\n'):
        row = row.split(',')
        if len(row) == 3:
            dists.append(float(row[1]))
            energies.append(float(row[2]))
    
    plt.title("Absorbed energy with distance")
    plt.xlabel("Distance")
    plt.ylabel("Normalized energy")
    plt.grid()
    plt.scatter(dists,energies)
    plt.savefig(f"{output_dir}/{results_filename}.png")

    f = open(f"{output_dir}/{results_filename}.dat", 'r')
    data = dict()
    for row in f.read().split('\n'):
        row = row.split(',')
        if len(row) != 3: continue
        if float(row[1]) in data.keys():
            data[float(row[1])].append(float(row[2]))
        else:
            data[float(row[1])] = [float(row[2])]

    scatter_dists = []
    scatter_energies = []
    for k,v in data.items():
        for e in v:
            scatter_dists.append(k)
            scatter_energies.append(e)

    plt.title("Absorbed energy with distance")
    plt.xlabel("Distance")
    plt.ylabel("Normalized energy")
    plt.grid()
    plt.scatter(scatter_energies, scatter_dists)
    plt.errorbar(
        data.keys(),
        [np.mean(points) for points in data.values()],
        [np.std(points) / math.sqrt(len(points)) for points in data.values()],
        0)
    plt.savefig(f"{output_dir}/{results_filename}.png")


if __name__ == "__main__":
    samples = []
    w = walk(samples_dir)
    for (dirpath, dirnames, filenames) in w:
        samples = [fn.strip(".flac") for fn in filenames]
        break # read only first level
    
    samples = ["m_d0_n2"]
    print(samples)

    results_filename = "oversaturated"
    with open(f"{output_dir}/{results_filename}.dat", 'w') as f:
        f.write("")
    f.close()

    for sample_filename in samples:
        threshold = 0.0 # .35
        cap = 1.0 # .95
        binc = 20

        sample_properties = sample_filename.split('_')
        distance = float(sample_properties[1][1:])
        # shield = float(sample_properties[2][1:])
        shield = 0.0
        
        process_data(Config(threshold, cap, binc, sample_filename, distance, shield))
        calculate_energy_absorption(sample_filename, results_filename)
    
    plot_results(results_filename)