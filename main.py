#!/usr/bin/env python3

import os
import wave

import pylab as lab

import numpy as np
from scipy.signal import butter, lfilter, freqz

path_sound_in = 'sound_in/'
path_sound_out = 'sound_out/'


def find_wav(path):
    path += '/' if path[-1] != '/' else ''
    wavs = []
    for fil in os.scandir(path):
        if fil.is_dir():
            wavs.extend(find_wav(path + fil.name))
        elif len(fil.name) > len('.wav') and fil.name[-len('.wav'):] == '.wav':
            wavs.append(path + fil.name)
    return sorted(wavs)


def read_wav(path, fs, samples):
    with wave.open(path, 'rb') as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        frames = wav.getnframes()
        fs = wav.getframerate()

        raw = [[] for i in range(channels)]

        for i in range(frames):
            frame = wav.readframes(1)
            for channel in range(channels):
                raw[channel].append(
                    int.from_bytes(
                        frame[channel * sample_width:channel * sample_width + sample_width],
                        byteorder='little',
                        signed=True
                    )
                )

        samples = np.array(raw) / 2**15
        if channels > 1:
            samples = sum(samples) / channels
        return fs, samples




def antialiasing_filter(lowcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = butter(order, low, btype='lowpass')
    return b, a

def oversampling(fin, fout, samples):
    step = int(round(fin / fout)) - 1
    return samples[::step]


def graf(s1, s2):
    lab.figure()
    lab.subplot(211)
    lab.plot(s1)
    lab.title('s1')

    lab.subplot(212)
    lab.plot(s2)
    lab.title('s2')


def cut_noise(samples):
    trash = 0.01
    for i in range(len(samples)):
        if abs(samples[i]) > trash:
            start = i
            break
    for i in range(len(samples) - 1, -1, -1):
        if abs(samples[i]) > trash:
            stop = i + 1
            break
    samples = samples[start:stop]
    samples = np.insert(samples, 0, 0)
    samples = np.insert(samples, len(samples), 0)
    return samples


if __name__ == '__main__':
    path_wav = find_wav(path_sound_in)
    size = 0
    size_edit = 0
    fo = 16e3
    for w in path_wav:
        fs, samples = read_wav(w)
        b, a = antialiasing_filter(0.5 * fo, fs, 13)
        samples_edit = lfilter(b, a, samples)
        samples_edit = oversampling(fs, fo, samples_edit)
        samples_edit = cut_noise(samples_edit)
        size += len(samples)
        size_edit += len(samples_edit)
        print('{}: {}\t{}'.format(w, len(samples), len(samples_edit)))

        #graf(samples, samples_edit)
        break
    print('size     : {:10d}'.format(size))
    print('size_edit: {:10d}'.format(size_edit))
    lab.show()
