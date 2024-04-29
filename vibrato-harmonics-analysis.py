import numpy as np
import scipy
import librosa, librosa.display
import matplotlib.pyplot as plt
import matplotlib.style as ms
import pandas as pd
from scipy.signal import butter, lfilter
from scipy.signal import freqz
ms.use("seaborn-v0_8")


if __name__ == '__main__': 
    
    sample_name = 'G3-circular'
    filepath = f'data-processed/{sample_name}/{sample_name}-sound-2.wav'
    sr = 44100 # sampling rate
    signal, sr = librosa.load(filepath, sr=sr, mono=False)
    print(signal.shape)
    print('{:2.3f}'.format(librosa.samples_to_time(signal.shape[0], sr=sr)))
    plt.figure(figsize=(10, 3))
    librosa.display.waveshow(y=signal, sr=sr, color="blue")
    
    
    f0, voicing, voicing_p = librosa.pyin(y=signal, sr=sr, fmin=200, fmax=700)
    S = np.abs(librosa.stft(signal))
    freqs = librosa.fft_frequencies(sr=sr)
    harmonics = np.arange(1, 13)
    f0_harm = librosa.f0_harmonics(S, freqs=freqs, f0=f0, harmonics=harmonics)
    
    fig, ax = plt.subplots(nrows=2, sharex=True)
    librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
                             x_axis='time', y_axis='log', ax=ax[0])
    times = librosa.times_like(f0)
    for h in harmonics:
        ax[0].plot(times, h * f0, label=f"{h}*f0")
    ax[0].legend(ncols=4, loc='lower right')
    ax[0].label_outer()
    librosa.display.specshow(librosa.amplitude_to_db(f0_harm, ref=np.max),
                             x_axis='time', ax=ax[1])
    ax[1].set_yticks(harmonics-1)
    ax[1].set_yticklabels(harmonics)
    ax[1].set(ylabel='Harmonics')
    
    
    def butter_bandpass(lowcut, highcut, fs, order=5):
        return butter(order, [lowcut, highcut], fs=fs, btype='band')
    
    def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
        b, a = butter_bandpass(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data)
        return y
    
    # Sample rate and desired cutoff frequencies (in Hz).
    fs = sr
    lowcut = 200 # filtering out background noise
    highcut = 20000
    
    # Plot the frequency response for a few different orders.
    y = butter_bandpass_filter(signal, lowcut, highcut, sr, order=6) # filtered signal
    S = np.abs(librosa.stft(y)) # spectrum
    fig, ax = plt.subplots(nrows=5, figsize=(8, 8))
    for order in [3, 6, 9]:
        b, a = butter_bandpass(lowcut, highcut, fs, order=order)
        w, h = freqz(b, a, fs=fs, worN=2000)
        ax[0].plot(w, abs(h), label="order = %d" % order)
    
    ax[0].plot([0, 0.5 * fs], [np.sqrt(0.5), np.sqrt(0.5)],
             '--', label='sqrt(0.5)')
    ax[0].set_xlabel('Frequency (Hz)')
    ax[0].set_ylabel('Gain')
    ax[0].grid(True)
    ax[0].legend(loc='best')
    S = np.abs(librosa.stft(signal)) # spectrum
    librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
                             x_axis='time', y_axis='log', ax=ax[1])
    S = np.abs(librosa.stft(y)) # spectrum
    librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
                             x_axis='time', y_axis='log', ax=ax[2])
    ax[3].plot(signal, label='Original signal')
    ax[3].plot(y)
    y_rms = librosa.feature.rms(y=y) # amplitude envelope
    ax[4].plot(y_rms[0])
    
    f0[np.argmax(y_rms[0])]
    k = 5 # average of the k frequencies with max amplitudes
    mean_f0 = np.mean(f0[np.argpartition(y_rms[0], len(y_rms[0]) - k)[-k:]])
    central_harmonics = harmonics * mean_f0
    
    
    # get samples from RMS
    times = librosa.times_like(y_rms)
    samples_indexes = librosa.time_to_samples(times)
    
    # filtering harmonics and extracting pitch and amplitude envelopes
    
    N_harmonics = 6
    # cut signal
    signal_harmonic = signal[samples_indexes[0]:samples_indexes[300]]
    filter_width = 50
    
    lowcut = central_harmonics[1] - filter_width # filtering out background noise
    highcut = central_harmonics[N_harmonics-1] + filter_width
    
    signal = signal[~np.isnan(signal)]
    signal = signal[signal != -np.inf]
    y = butter_bandpass_filter(signal, lowcut, highcut, sr, order=6) # filtered signal
    y = y[~np.isnan(y)]
    y = y[y != -np.inf]
    S = np.abs(librosa.stft(y)) # spectrum
    
    fig, ax = plt.subplots(nrows=6, figsize=(10, 16))
    b, a = butter_bandpass(lowcut, highcut, fs, order=3)
    w, h = freqz(b, a, fs=fs, worN=2000)
    ax[0].plot(w, abs(h))
    
    ax[0].plot([0, 0.5 * fs], [np.sqrt(0.5), np.sqrt(0.5)], '--', label='sqrt(0.5)')
    ax[0].set_xlabel('Frequency (Hz)')
    ax[0].set_ylabel('Gain')
    ax[0].grid(True)
    ax[0].legend(loc='best')
    S = np.abs(librosa.stft(signal)) # spectrum
    librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max), x_axis='time', y_axis='log', ax=ax[1])
    #S = np.abs(librosa.stft(y)) # spectrum
    librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max), x_axis='time', y_axis='log', ax=ax[2])
    ax[3].plot(signal, label='Original signal')
    ax[3].plot(y)
    
    harms = range(N_harmonics)
    for harm in harms:
    
        harmonic = harm
        
        # Sample rate and desired cutoff frequencies (in Hz).
        fs = sr
        lowcut = central_harmonics[harmonic] - filter_width
        highcut = central_harmonics[harmonic] + filter_width
        
        # Filter signal
        y = butter_bandpass_filter(signal_harmonic, lowcut, highcut, sr, order=3) # filtered signal
        y = y[~np.isnan(y)]
        y = y[y != -np.inf]
        S = np.abs(librosa.stft(y)) # spectrum
        y_freq = librosa.yin(y, sr=sr, fmin=50, fmax=880, frame_length=2048) # central frequency 
        y_rms = librosa.feature.rms(y=y) # amplitude envelope
    
        # remove outliers by interpolation with the rolling average
        df = pd.DataFrame({'Data':y_freq[20:170]})
        #df = pd.DataFrame({'Data':y_freq})
        r = df.rolling(window=10)
        mps_up, mps_low = r.mean() + 3 * r.std(), r.mean()  -  3 * r.std()
        df.loc[~df['Data'].between(mps_low.Data, mps_up.Data), 'Data'] = np.NaN
        df['Data'] = df['Data'].bfill()
        y_freq_clean = (df['Data'].values - df['Data'].values.mean()) / df['Data'].values.std()
        
        ax[4].plot(y_freq_clean, label=f'{central_harmonics[harmonic]:.2f} Hz')
        ax[4].legend(loc='upper right')
    
        ax[5].plot(y_rms[0], label=f'{central_harmonics[harmonic]:.2f} Hz')
        ax[5].legend(loc='upper right')
        
        b, a = butter_bandpass(lowcut, highcut, fs, order=3)
        w, h = freqz(b, a, fs=fs, worN=2000)
        ax[0].plot(w, abs(h), label=f'{central_harmonics[harmonic]:.2f} Hz')
        ax[0].plot([0, 0.5 * fs], [np.sqrt(0.5), np.sqrt(0.5)], '--')
        ax[0].set_xlabel('Frequency (Hz)')
        ax[0].set_ylabel('Gain')
        ax[0].grid(True)
        ax[0].legend(loc='upper right')
    
    ax[0].set_title('Filters')
    ax[1].set_title('Spectrogram before filtering')
    ax[2].set_title('Spectrogram after filtering')
    ax[3].set_title('Signals before and after filtering')
    ax[4].set_title('Pitch oscillations of the different harmonics')
    ax[5].set_title('Amplitude envelopes of the different harmonics')
    
    fig.tight_layout(pad=2.0)
    fig.suptitle('Data analysis')
    plt.show()
