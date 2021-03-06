#!/usr/bin/env python3

import os

import numpy as np
from scipy.signal import butter, lfilter, freqz
from scipy.io import wavfile

path_sound_in = 'sound'

def find_wav(path):
    path += '/' if path[-1] != '/' else ''
    wavs = []
    for fil in os.scandir(path):
        if fil.is_dir():
            wavs.extend(find_wav(path + fil.name))
        elif len(fil.name) > len('.wav') and fil.name[-len('.wav'):] == '.wav' and not ('_edit' in fil.name):
            wavs.append(path + fil.name)
    return sorted(wavs)


def read_wav(path):
    fs, raw = wavfile.read(path)
    dtype = raw.dtype
    raw  = np.array(raw, np.float)

    if len(raw.shape) == 2:
        channels = raw.shape[-1]
        raw = sum(raw.transpose()) / channels

    if dtype == np.dtype(np.int16):
        raw /= 2**15
    elif dtype == np.dtype(np.uint8):
        raw -= 2**7 - 1
        raw /= 2**7 - 1

    return fs, raw

def write_wav(path, fs, dtype, samples):
    if dtype == 12:
        samples = np.array(samples * (2**15 - 1), np.int16)
        samples = np.right_shift(samples, 4)
        samples = np.left_shift(samples, 4)

    elif dtype == 16:
        samples = np.array(samples * (2**15 - 1), np.int16)

    wavfile.write(path, int(fs), samples)


def antialiasing_filter(lowcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = butter(order, low, btype='lowpass')
    return b, a

def oversampling(fin, fout, samples):
    step = int(round(fin / fout))
    return samples[::step]


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


def create_h(path, dtype, samples):
    name = path.replace('/', '_')[:-4]
    h_file = '#define {}_SIZE {}\n'.format(name.upper(), len(samples))
    h_file += 'extern const '
    h_file += 'uint16_t ' if dtype > 8 else 'uint8_t '
    h_file += name
    h_file += '[{}_SIZE];\n'.format(name.upper())

    return h_file

def sort_h(header):
    defines = []
    externs = []
    for line in header:
        if 'extern' in line:
            externs.append(line)
        elif '#define' in line:
            defines.append(line)

    return defines + ['\n\n'] + externs

def create_c(path, dtype, samples):
    name = path.replace('/', '_')[:-4]
    c_file = '\nconst '
    c_file += 'uint16_t ' if dtype > 8 else 'uint8_t '
    c_file += name
    c_file += '[{}_SIZE] = '.format(name.upper())
    c_file += '{'

    if dtype == 8:
        samples *= 2**7 - 1
        samples += 2**7 - 1
    elif dtype == 12:
        samples *= 2**11 - 1
        samples += 2**11 - 1
    elif dtype == 16:
        samples *= 2**15 - 1
        samples += 2**15 - 1

    samples = np.array(samples, np.uint16)
    for s in samples:
        c_file += '{},'.format(s)

    c_file = c_file[:-1]
    c_file += '};\n'

    return c_file


if __name__ == '__main__':
    path_wav = find_wav(path_sound_in)
    size = 0
    size_edit = 0
    fo = 16e3
    c_file = ''
    h_file = ''
    for w in path_wav:
        fs, samples = read_wav(w)
        b, a = antialiasing_filter(0.5 * fo, fs, 2)
        samples_edit = lfilter(b, a, samples)
        samples_edit = oversampling(fs, fo, samples_edit)
        samples_edit = cut_noise(samples_edit)
        path_edit = ''.join((w[:-4], '_edit.wav'))
        write_wav(path_edit, fo, 12, samples_edit)
        c_file += create_c(w, 12, samples_edit)
        h_file += create_h(w, 12, samples_edit)
        size += len(samples)
        size_edit += len(samples_edit)
        print('{}: {:10d}     edit {:10d}'.format(w, len(samples), len(samples_edit)))


    print('===================================================================')
    print('size     : {:10,d} samples -> {:10,d} B'.format(size, 2 * size))
    print('size_edit: {:10,d} samples -> {:10,d} B'.format(size_edit, 2 * size_edit))
    print('compresion: {:9,d} %'.format(100 - int(round(size_edit/(size/100)))))

    with open('sound.c', 'w') as fw:
        fw.write(c_file)

    with open('sound.h', 'w') as fw:
        fw.write(h_file)

    with open('sound.h', 'r') as fr:
        header = fr.readlines()

    with open('sound.h', 'w') as fw:
        fw.writelines(sort_h(header))
