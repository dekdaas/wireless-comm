import sounddevice as sd
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig
import scipy
from skimage.feature import match_template
from textparser import parse
from textparser import printToFile
from reed import RSCodec
import time

#######################################################################################
#  _    _                                     _           _
# | |  | |                                   | |         (_)
# | |  | | __ ___   _____    __ _ _ __   __ _| |_   _ ___ _ ___
# | |/\| |/ _` \ \ / / _ \  / _` | '_ \ / _` | | | | / __| / __|
# \  /\  / (_| |\ V /  __/ | (_| | | | | (_| | | |_| \__ \ \__ \
#  \/  \/ \__,_| \_/ \___|  \__,_|_| |_|\__,_|_|\__, |___/_|___/
#                                                __/ |
#                                               |___/
#######################################################################################
def strip_wave(wave, template, head=True, noise=False, off_one=0):
    """
    Separates wave into two waves (information bearing & noise) by matching template to wave and splitting at
    the point of best match/overlap (determined by normalized cross-correlation)

    Input wave is modeled as
    ______________________________________
    | noise | head | wave | foot | noise |
    |_______|______|______|______|_______|
    Parameters:
        head: Indicates that the template is head. If noise is False and head is:
              True -> template is head. strip_wave will return
                wave = ______________________   noise = ________________
                       | wave | foot | noise |          | noise | head |
                       |______|______|_______|          |_______|______|

              Otherwise, template is foot. strip_wave will return
                wave = ______________________   noise = ________________
                       | noise | head | wave |          | foot | noise |
                       |_______|______|______|          |______|_______|
        noise: Indicates that noise should be stripped instead; as a result, noise will not contain head or foot

    """
    #print(template.size, wave.size)
    assert(template.size <= wave.size)
    result = match_template(np.reshape(wave, [wave.size, 1]), np.reshape(template, [template.size, 1]))

    # print(result)
    #plt.plot(result)
    #plt.show()
    template_start = np.argmax(result)
    template_size = template.size

    if noise:
        template_size = 0
    if head: # remove header from information bearing signal
        noise = wave[:template_start + template_size]
        wave = wave[template_start + template_size:]
    else: # remove footer from information bearing signal
        noise = wave[template_start + off_one:]
        wave = wave[:template_start + off_one]
    return wave, noise

def demodulate(wave, strip=True, off_one=0):
    """ Provide time-series of characters heard in a wave"""
    """ Doesn't quite work; this is because there is not an identity mapping from frequency interval to character. Use to figure out how to modulate characters to frequencies, and tune to frequency/time resolution"""
    # align wave to start at header. calibrate alphabet from header
    if strip:
        wave, head_noise = strip_wave(wave, make_header(), off_one=off_one)
        #print('strip', wave.size, head_noise.size)
        wave, foot_noise = strip_wave(wave, make_footer(), head=False, off_one=off_one)

    # find dominant frequency interval per time interval
    f, t, Sxx = spec(wave)

    # drop out frequency intervals where alphabet does not reside. allows immediate application of chr() to find character
    # only works for current hand tuned parameters: nsperseg/duration, start/ratio
    # the first pair adjusts time buckets to contain single note.
    # the second adjusts range of frequency buckets each note falls into (want one bucket per note)
    # nsperseg = 2205, duration = 0.2, start = 100, ratio = 120
    f, Sxx = f[5::6], Sxx[5::6]
    #print(f.size, Sxx.size)
    dominant = []

    assert(Sxx.T[0].size == f.size)
    for spectrum in Sxx.T:
        dominant.append(chr(np.argmax(spectrum)))
    return dominant

#######################################################################################
#  _____                       _   _   _     _
# /  ___|                     | | | | | |   (_)
# \ `--.  ___  _   _ _ __   __| | | |_| |__  _ _ __   __ _ ___
#  `--. \/ _ \| | | | '_ \ / _` | | __| '_ \| | '_ \ / _` / __|
# /\__/ / (_) | |_| | | | | (_| | | |_| | | | | | | | (_| \__ \
# \____/ \___/ \__,_|_| |_|\__,_|  \__|_| |_|_|_| |_|\__, |___/
#                                                     __/ |
#                                                    |___/
#######################################################################################
ascii_to_freq = {}
freq_to_ascii = {}

start = 100 #27.5/((2**(1/12.))**5)
ratio = 120#1.05 #2 ** (1/12.)
ascii = range(0, 128)

for ascii_number in ascii:
	ascii_to_freq[chr(ascii_number)] = start + (ratio * (ascii_number))
	# print start * (ratio ** ascii_number)
	freq_to_ascii[start + (ratio * ascii_number)] = chr(ascii_number)

def make_sound(text, duration, amplitude, fs=44100):
    final_wave = np.array([]) #make_header()
    for char in text:
        #print(char, ascii_to_freq[char], end=' ')
        t = np.arange(np.ceil(duration * fs)) / fs
        #sig = amplitude * np.sin(2 * np.pi * ascii_to_freq[char] * t)
        sig = sine_wave(ascii_to_freq[char], duration, fs, amplitude)
        #final_wave = append_waves(final_wave, np.zeros(int(0.025*duration * fs)), sig, np.zeros(int(0.025*duration * fs)))
        final_wave = append_waves(final_wave, sig)
    return final_wave

def add_waves(*waves):
    """ Take array of input amplitude waves and add and normalize.
    Parameters
    ----------
    waves : array_like
        Audio data to be added together.
    POSSIBLE: have input array of offsets per wave, to tell where in final wave to add wave
    """
    if len(waves) == 0:
        return -1
    final_wave = np.zeros(max([len(wave) for wave in waves]))
    for wave in waves:
        final_wave[:len(wave)] += wave
    final_wave /= max(final_wave)
    return final_wave

def append_waves(*waves):
    final_wave= np.array([])
    for wave in waves:
        final_wave = np.hstack((final_wave, wave))
    return final_wave

def make_header():
    #return append_waves(sine_wave(550, 0.2), sine_wave(850, 0.4), sine_wave(1700, 0.2), sine_wave(700, 0.2))
    # s = ''
    # for i in list(ascii)[::30]:
    #     s += chr(i)
    # return make_sound(s, 0.2, 1)
    a = add_waves(square_wave(1000, 0.2), square_wave(19000, 0.2), square_wave(2000, 0.2), sine_wave(14000, 0.2))
    b = add_waves(sine_wave(13208, 0.2), square_wave(11698, 0.2), square_wave(10272, 0.2), sine_wave(12310, 0.2))
    return append_waves(b, b, a)
def make_footer():
    #return append_waves(sine_wave(550, 0.2), sine_wave(850, 0.4), sine_wave(1700, 0.2), sine_wave(700, 0.2))
    # s = ''
    # for i in list(ascii)[::30]:#range(0, 128, 30):
    #     s += chr(i)
    # return make_sound(s, 0.2, 1)
    a = add_waves(square_wave(12900, 0.2), square_wave(4630, 0.2), square_wave(11490, 0.2), sine_wave(18230, 0.2))
    b = add_waves(square_wave(9173, 0.2), square_wave(16198, 0.2), square_wave(13272, 0.2), sine_wave(2310, 0.2))
    return append_waves(a, b, a)

def sine_wave(freq, duration, fs=44100, amplitude=1):
    x = np.arange(np.ceil(duration * fs)) / fs
    return np.sin(freq * 2*np.pi*x) * amplitude

def square_wave(freq, duration, sf=44100, amplitude=1):
    """ Returns a square wave oscillating between 1 and -1 with an input frequency.
    Parameters
    ----------
    freq: float
       Frequency in hertz of desired wave
    duration: float
       Length of wave in seconds
    sf: int
        Samples per second
    amplitude: height of wave (volume)
    """
    #wavelength = int(sf*duration/freq) # wavelength
    wavelength = int(sf/freq) # wavelength
    wave = np.zeros(int(sf*duration))

    waveform = amplitude * np.ones(wavelength)
    waveform[wavelength//2:] *= -1 # oscillate
    for i in range(int(freq*duration)):
        wave[i*wavelength : (i+1)*wavelength] = waveform
    return wave

def rec(duration = 1, sf = 44100):
    myrec = sd.rec(duration, sf, 1)
    return scipy.mean(myrec, axis=1)


def play(freq, duration=1, sf=44100):
    """ Test and view waves produced
    """
    wave = sin_wave(freq, sf, duration)
    sd.play(wave, sf)

#######################################################################################
# ______ _       _   _   _
# | ___ \ |     | | | | (_)
# | |_/ / | ___ | |_| |_ _ _ __   __ _
# |  __/| |/ _ \| __| __| | '_ \ / _` |
# | |   | | (_) | |_| |_| | | | | (_| |
# \_|   |_|\___/ \__|\__|_|_| |_|\__, |
#                                 __/ |
#                                |___/
#######################################################################################

def spec(wave):
    return sig.spectrogram(wave, nperseg=2205, noverlap=0)

def plot_spec(wave):
    """ Plots the frequency spectrum of the input wave"""
    f, t, Sxx = spec(wave)
    plt.pcolormesh(t, f, Sxx)
    plt.show()
    return f, t, Sxx

def plot_recording(duration=1, sf=44100):
    """ Records an audio signal and plots its frequency spectrum """
    sample_points = sf*duration
    myrec = sd.rec(sample_points, sf, 1)
    sd.wait()
    f, t, Sxx = plot_spec(np.reshape(myrec, [myrec.size]))
    return myrec, f, t, Sxx

###############################################################################
# _____         _   _               _   _     _
#|_   _|       | | (_)             | | | |   (_)
#  | | ___  ___| |_ _ _ __   __ _  | |_| |__  _ _ __   __ _ ___
#  | |/ _ \/ __| __| | '_ \ / _` | | __| '_ \| | '_ \ / _` / __|
#  | |  __/\__ \ |_| | | | | (_| | | |_| | | | | | | | (_| \__ \
#  \_/\___||___/\__|_|_| |_|\__, |  \__|_| |_|_|_| |_|\__, |___/
#                            __/ |                     __/ |
#                           |___/                     |___/
###############################################################################
# Example usage
# play(440) # A4
# chord = add_waves([square_wave(440, 2), square_wave(523.25, 2), square_wave(659.25, 2)])
# sd.play(chord)

dur = 0.05  # duration in seconds
amp = 1  # amplitude (full scale: +-1.0)
freq = 1000.  # frequency in Hertz
sd.default.samplerate = 44100  # sampling frequency in Hertz

s = ''
for n in ascii:
   s += chr(n)
w = make_sound(s, dur, amp)
ws = append_waves(make_header(), w, make_footer())
#ws = make_sound('abcdefghijklmnopqrst', dur, amp)

def test():
    test = append_waves(sine_wave(0, 0.5), sine_wave(1000, 0.5), make_header(), sine_wave(500, 1))
    test /= 5
    myrec = sd.playrec(test, 44100, 1)
    sd.wait()
    wave, noise = strip_wave(myrec, make_header())
    head, noise = strip_wave(noise, make_header(), noise=True)
    return noise, head, wave # expect head.size = wave.size = noise.size = 44100

def test3():
    myrec = sd.playrec(w, 44100, 1)
    myrec = np.reshape(myrec, [myrec.size])
    sd.wait()
    #plot_spec(myrec)
    decrypt = demodulate(myrec, False)
    right = [d == char for d, char in zip(decrypt, demodulate(w, False))]
    print(decrypt, right.count(True), right.count(True)/len(right) )
    print(demodulate(w, False))

def test2():
    myrec = sd.playrec(ws, 44100, 1)
    myrec = np.reshape(myrec, [myrec.size])
    sd.wait()
    #plot_spec(myrec)
    decrypt = demodulate(myrec)
    right = [d == w for d, w in zip(decrypt, demodulate(w))]
    print(decrypt, right.count(True), right.count(True)/len(right) )
    print(demodulate(w, False))

def maxfreq(s):
    dic = {}
    for i in s:
        if i not in dic:
            dic[i] = 1
        else:
            dic[i] += 1
    maxi = 0
    arg = None
    for i in dic:
        if dic[i] > maxi:
            maxi = dic[i]
            arg = i
    return arg

rs = RSCodec(50)

def encrypt(text):
    a = rs.encode(text)
    return a.decode("utf-8")


def decode(filt):
    s = ""
    for i in filt:
        s += i
    #print(s)
    #print(len(s))
    b = bytearray()
    b.extend(map(ord,s))
    #print(b)
    retstr = rs.decode(b)
    #print(retstr)
    return retstr.decode("utf-8")

strin = "It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness, it was the epoch of belief, it was the epoch of incredulity, it was the season of Light, it was the season of Darkness, it was the spring of hope, it was the winter of despair, we had everything before us, we had nothing before us, we were all going direct to Heaven, we were all going direct the other way - in short, the period was so far like the present period, that some of its noisiest" #authorities insisted on its being received, for good or for evil, in the superlative degree of comparison only."

def test4(txt='oopsie daisy :O'):
    start_time = time.time()
    text = encrypt(strin)
    #print("encrypted to " + text)
    #print("encryption len: " + str(len(text)))
    w = make_sound(text, 0.05, 1)
    w = append_waves(sine_wave(0, 1), make_header(),w,make_footer(), sine_wave(0, 1))
    myrec = sd.playrec(w, 44100, 1)
    myrec = np.reshape(myrec, [myrec.size])
    sd.wait()
    #plot_spec(myrec)
    decrypt = demodulate(myrec)
    if len(decrypt) != len(text):
        print("Off by one")
        decrypt = demodulate(myrec, True, 1100)
    decode(decrypt)
    # filtered = []
    # for i in range(len(decrypt)//4):
    #     if 4*i + 4 < len(decrypt):
    #         filtered.append(maxfreq(decrypt[4*i:4*i + 4]))
    #     else:
    #         filtered.append(maxfreq(decrypt[4*i:]))
    # print(filtered, len(filtered))
    # print(decode(filtered))
    run_time = (time.time() - start_time)
    print("---- %s seconds ----" % run_time)
    print("Bit rate: %s" % str(8*len(strin) / run_time))
    print("Physical Bit rate: %s" % str(8*len(text) / run_time))
    
    print(decode(decrypt))
    return myrec

########################
#   ___  ______ _____  #
#  / _ \ | ___ \_   _| #
# / /_\ \| |_/ / | |   #
# |  _  ||  __/  | |   #
# | | | || |    _| |_  #
# \_| |_/\_|    \___/  #
######################## 
 
def receive(duration = 8, file_name = 'received.txt'):
    try:
        assert(type(file_name) = type(''))
    except:
        print("file_name must be a string")
        return

    start_time = time.time()
    myrec = sd.rec(44100*duration, 44100, 1)
    sd.wait()
    print(myrec.size)
    myrec = np.reshape(myrec, [myrec.size])
    decrypt = demodulate(myrec)
    print(''.join(decrypt))

    dec = decode(decrypt)
    run_time = (time.time() - start_time)
    print("---- %s seconds ----" % run_time)

    print(dec)
    printToFile(dec, file_name)

def transmit(filename):
    text = parse(filename)
    txt = make_sound(encrypt(text), 0.05, 1)
    txt = append_waves(make_header(), txt, make_footer())
    sd.play(txt, 44100, 1)
    

