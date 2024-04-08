import numpy as np
import scipy
import librosa, librosa.display
import matplotlib.pyplot as plt
import matplotlib.style as ms
import os 
import scipy.io.wavfile as wavf
ms.use("seaborn-v0_8")  

if __name__ == '__main__': 
    
    path = 'data'
    files = os.listdir(path)
    files = [file for file in files if file.endswith('.wav')]
    filenames = ['-'.join(file.split('.')[0].split('-')[:-1]) for file in files]
    
    
    filename = 'G4-vertical'
    plot = True
    
    for filename in filenames:
        
        print(f'Segmenting {filename}...')
        
        # save dir
        save_path = f'data-processed/{filename}'
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        # load sound
        sr = 44100 # sampling rate
        file = f'data/{filename}-sound.wav'
        signal, sr = librosa.load(file, sr=sr, mono=False)
        print(signal.shape)
        print('{:2.3f}'.format(librosa.samples_to_time(signal.shape[0], sr=sr)))
        plt.figure(figsize=(10, 3))
        if plot:
            librosa.display.waveshow(y=signal, sr=sr, color="blue")
        
        # load sensor data
        f = open(f"data/{filename}-acc.hand.x.txt", "r")
        acc_hand_x = f.read()
        acc_hand_x = acc_hand_x.split(" ")
        acc_hand_x = np.array([float(x) for x in acc_hand_x])
        
        f = open(f"data/{filename}-acc.hand.y.txt", "r")
        acc_hand_y = f.read()
        acc_hand_y = acc_hand_y.split(" ")
        acc_hand_y = np.array([float(x) for x in acc_hand_y])
        
        f = open(f"data/{filename}-acc.hand.z.txt", "r")
        acc_hand_z = f.read()
        acc_hand_z = acc_hand_z.split(" ")
        acc_hand_z = np.array([float(x) for x in acc_hand_z])
        
        f = open(f"data/{filename}-gyro.hand.x.txt", "r")
        gyro_hand_x = f.read()
        gyro_hand_x = gyro_hand_x.split(" ")
        gyro_hand_x = np.array([float(x) for x in gyro_hand_x])
        
        f = open(f"data/{filename}-gyro.hand.y.txt", "r")
        gyro_hand_y = f.read()
        gyro_hand_y = gyro_hand_y.split(" ")
        gyro_hand_y = np.array([float(x) for x in gyro_hand_y])
        
        f = open(f"data/{filename}-gyro.hand.z.txt", "r")
        gyro_hand_z = f.read()
        gyro_hand_z = gyro_hand_z.split(" ")
        gyro_hand_z = np.array([float(x) for x in gyro_hand_z])
        
        if plot:
            fig, ax = plt.subplots(3, 2, figsize=(10,5), sharex=True)
            ax[0, 0].plot(acc_hand_x)
            ax[1, 0].plot(acc_hand_y)
            ax[2, 0].plot(acc_hand_z)
            ax[0, 1].plot(gyro_hand_x)
            ax[1, 1].plot(gyro_hand_y)
            ax[2, 1].plot(gyro_hand_z)
        
            ax[0, 0].set(title='Accelerometer data')
            ax[0, 1].set(title='Gyroscope data')
        
        # identify onsets
        o_env = librosa.onset.onset_strength(y=signal, sr=sr)
        times = librosa.times_like(o_env, sr=sr)
        onset_frames = librosa.onset.onset_detect(onset_envelope=o_env, sr=sr, backtrack=True)
        peaks = librosa.util.peak_pick(o_env, pre_max=500, post_max=500, 
                                       pre_avg=500, post_avg=700, delta=0.5, wait=10)
        
        S = np.abs(librosa.stft(y=signal))
        rms = librosa.feature.rms(S=S)
        onset_bt_rms = librosa.onset.onset_backtrack(peaks, rms[0])
        
        if plot:
            fig, ax = plt.subplots(nrows=2, sharex=True)
            librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
                                     y_axis='log', x_axis='time', ax=ax[0])
            ax[0].label_outer()
            ax[1].plot(times, rms[0], label='RMS')
            ax[1].vlines(times[onset_bt_rms], 0, rms.max(), label='Backtracked (RMS)', color='r')
            ax[1].legend()
        
        # transfer onset splits to sensor data
        signal_duration = signal.shape[0] / sr
        # interval at which a new sensor data is received
        sensor_interval = signal_duration / acc_hand_x.shape[0]
        sound2sensor = acc_hand_x.shape[0] / signal.shape[0]
        sensor_peaks = librosa.frames_to_samples(onset_bt_rms) * sound2sensor
        
        if plot:
            fig, ax = plt.subplots(nrows=7, figsize=(10,10))
            ax[0].plot(signal, label='signal')
            ax[0].vlines(librosa.frames_to_samples(onset_bt_rms), signal.min(), signal.max(), 
                       color='r', alpha=0.9, linestyle='--', label='Onsets')
            ax[0].legend()
            ax[1].plot(acc_hand_x, label='accelerometer')
            ax[1].vlines(sensor_peaks, acc_hand_x.min(), acc_hand_x.max(), 
                       color='r', alpha=0.9, linestyle='--', label='Onsets')
            ax[2].plot(acc_hand_y, label='accelerometer')
            ax[2].vlines(sensor_peaks, acc_hand_y.min(), acc_hand_y.max(), 
                       color='r', alpha=0.9, linestyle='--', label='Onsets')
            ax[3].plot(acc_hand_z, label='accelerometer')
            ax[3].vlines(sensor_peaks, acc_hand_z.min(), acc_hand_z.max(), 
                       color='r', alpha=0.9, linestyle='--', label='Onsets')
            
            ax[4].plot(gyro_hand_x, label='accelerometer')
            ax[4].vlines(sensor_peaks, gyro_hand_x.min(), gyro_hand_x.max(), 
                       color='r', alpha=0.9, linestyle='--', label='Onsets')
            ax[5].plot(gyro_hand_y, label='accelerometer')
            ax[5].vlines(sensor_peaks, gyro_hand_y.min(), gyro_hand_y.max(), 
                       color='r', alpha=0.9, linestyle='--', label='Onsets')
            ax[6].plot(gyro_hand_z, label='accelerometer')
            ax[6].vlines(sensor_peaks, gyro_hand_z.min(), gyro_hand_z.max(), 
                       color='r', alpha=0.9, linestyle='--', label='Onsets')
            
            fig.suptitle(f'{filename} onset split', fontsize=15)
            filepath = os.path.join(save_path, 'split.png')
            fig.savefig(filepath)
        
        # split tracks
        peak_samples = librosa.frames_to_samples(onset_bt_rms)
        segments = []
        previous_peak = 0
        for peak in peak_samples:
            segment = signal[previous_peak:peak]
            segments.append(segment)
            previous_peak = peak
        segments.append(signal[previous_peak:])
        audio_segments = segments[1:]
    
        sensor_peaks_int = [int(x) for x in sensor_peaks]
        sensor_segments = []
        previous_peak = 0
        for peak in sensor_peaks_int:
            segment = acc_hand_x[previous_peak:peak]
            sensor_segments.append(segment)
            previous_peak = peak
        sensor_segments.append(acc_hand_x[previous_peak:])
        acc_hand_x_segments = sensor_segments[1:]
        
        sensor_segments = []
        previous_peak = 0
        for peak in sensor_peaks_int:
            segment = acc_hand_y[previous_peak:peak]
            sensor_segments.append(segment)
            previous_peak = peak
        sensor_segments.append(acc_hand_x[previous_peak:])
        acc_hand_y_segments = sensor_segments[1:]
        
        sensor_segments = []
        previous_peak = 0
        for peak in sensor_peaks_int:
            segment = acc_hand_z[previous_peak:peak]
            sensor_segments.append(segment)
            previous_peak = peak
        sensor_segments.append(acc_hand_x[previous_peak:])
        acc_hand_z_segments = sensor_segments[1:]
        
        sensor_segments = []
        previous_peak = 0
        for peak in sensor_peaks_int:
            segment = gyro_hand_x[previous_peak:peak]
            sensor_segments.append(segment)
            previous_peak = peak
        sensor_segments.append(acc_hand_x[previous_peak:])
        gyro_hand_x_segments = sensor_segments[1:]
        
        sensor_segments = []
        previous_peak = 0
        for peak in sensor_peaks_int:
            segment = gyro_hand_y[previous_peak:peak]
            sensor_segments.append(segment)
            previous_peak = peak
        sensor_segments.append(acc_hand_x[previous_peak:])
        gyro_hand_y_segments = sensor_segments[1:]
        
        sensor_segments = []
        previous_peak = 0
        for peak in sensor_peaks_int:
            segment = gyro_hand_z[previous_peak:peak]
            sensor_segments.append(segment)
            previous_peak = peak
        sensor_segments.append(acc_hand_x[previous_peak:])
        gyro_hand_z_segments = sensor_segments[1:]
                
        for i in range(len(audio_segments)):
            out_f = f'{filename}-sound-{i}.wav'
            filepath = os.path.join(save_path, out_f)
            wavf.write(filepath, sr, audio_segments[i])
        
            sensor_filename = f'{filename}-sensors-{i}.npy'
            filepath = os.path.join(save_path, sensor_filename)
            sensor_data = []
            sensor_data.append(acc_hand_x_segments[i])
            sensor_data.append(acc_hand_y_segments[i])
            sensor_data.append(acc_hand_z_segments[i])
            sensor_data.append(gyro_hand_x_segments[i])
            sensor_data.append(gyro_hand_y_segments[i])
            sensor_data.append(gyro_hand_z_segments[i])
            sensor_data = np.array(sensor_data)
            np.save(filepath, sensor_data)
        
