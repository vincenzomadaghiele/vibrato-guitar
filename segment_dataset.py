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
    
    
    sr = 44100 # sampling rate
    plot = True
    
    for sample_name in filenames:
        
        print(f'Segmenting {sample_name}...')
        
        file = f'data/{sample_name}-sound.wav'
        signal, sr = librosa.load(file, sr=sr, mono=False)
        print(signal.shape)
        print('{:2.3f}'.format(librosa.samples_to_time(signal.shape[0], sr=sr)))
        plt.figure(figsize=(10, 3))
        librosa.display.waveshow(y=signal, sr=sr, color="blue")
        
        fig, ax = plt.subplots(3, 2, figsize=(10,5), sharex=True)
        
        f = open(f"data/{sample_name}-acc.left.x.txt", "r")
        acc_left_x = f.read()
        acc_left_x = acc_left_x.split(" ")
        acc_left_x = np.array([float(x) for x in acc_left_x])
        ax[0, 0].plot(acc_left_x)
        
        f = open(f"data/{sample_name}-acc.left.y.txt", "r")
        acc_left_y = f.read()
        acc_left_y = acc_left_y.split(" ")
        acc_left_y = np.array([float(x) for x in acc_left_y])
        ax[1, 0].plot(acc_left_y)
        
        f = open(f"data/{sample_name}-acc.left.z.txt", "r")
        acc_left_z = f.read()
        acc_left_z = acc_left_z.split(" ")
        acc_left_z = np.array([float(x) for x in acc_left_z])
        ax[2, 0].plot(acc_left_z)
        
        f = open(f"data/{sample_name}-gyro.left.x.txt", "r")
        gyro_left_x = f.read()
        gyro_left_x = gyro_left_x.split(" ")
        gyro_left_x = np.array([float(x) for x in gyro_left_x])
        ax[0, 1].plot(gyro_left_x)
        
        f = open(f"data/{sample_name}-gyro.left.y.txt", "r")
        gyro_left_y = f.read()
        gyro_left_y = gyro_left_y.split(" ")
        gyro_left_y = np.array([float(x) for x in gyro_left_y])
        ax[1, 1].plot(gyro_left_y)
        
        f = open(f"data/{sample_name}-gyro.left.z.txt", "r")
        gyro_left_z = f.read()
        gyro_left_z = gyro_left_z.split(" ")
        gyro_left_z = np.array([float(x) for x in gyro_left_z])
        ax[2, 1].plot(gyro_left_z)
        
        ax[0, 0].set(title='Accelerometer data')
        ax[0, 1].set(title='Gyroscope data')
        fig.suptitle("Left hand")
        plt.show()
        '''
        fig, ax = plt.subplots(3, 2, figsize=(10,5), sharex=True)
        
        f = open(f"data/{sample_name}-acc.right.x.txt", "r")
        acc_right_x = f.read()
        acc_right_x = acc_right_x.split(" ")
        acc_right_x = np.array([float(x) for x in acc_right_x])
        ax[0, 0].plot(acc_right_x)
        
        f = open(f"data/{sample_name}-acc.right.y.txt", "r")
        acc_right_y = f.read()
        acc_right_y = acc_right_y.split(" ")
        acc_right_y = np.array([float(x) for x in acc_right_y])
        ax[1, 0].plot(acc_right_y)
        
        f = open(f"data/{sample_name}-acc.right.z.txt", "r")
        acc_right_z = f.read()
        acc_right_z = acc_right_z.split(" ")
        acc_right_z = np.array([float(x) for x in acc_right_z])
        ax[2, 0].plot(acc_right_z)
        
        f = open(f"data/{sample_name}-gyro.right.x.txt", "r")
        gyro_right_x = f.read()
        gyro_right_x = gyro_right_x.split(" ")
        gyro_right_x = np.array([float(x) for x in gyro_right_x])
        ax[0, 1].plot(gyro_right_x)
        
        f = open(f"data/{sample_name}-gyro.right.y.txt", "r")
        gyro_right_y = f.read()
        gyro_right_y = gyro_right_y.split(" ")
        gyro_right_y = np.array([float(x) for x in gyro_right_y])
        ax[1, 1].plot(gyro_right_y)
        
        f = open(f"data/{sample_name}-gyro.right.z.txt", "r")
        gyro_right_z = f.read()
        gyro_right_z = gyro_right_z.split(" ")
        gyro_right_z = np.array([float(x) for x in gyro_right_z])
        ax[2, 1].plot(gyro_right_z)
        
        ax[0, 0].set(title='Accelerometer data')
        ax[0, 1].set(title='Gyroscope data')
        fig.suptitle("Right hand")
        plt.show()
        '''
        
        
        o_env = librosa.onset.onset_strength(y=signal, sr=sr)
        times = librosa.times_like(o_env, sr=sr)
        onset_frames = librosa.onset.onset_detect(onset_envelope=o_env, sr=sr, backtrack=True)
        peaks = librosa.util.peak_pick(o_env, pre_max=500, post_max=500, 
                                       pre_avg=500, post_avg=700, delta=0.5, wait=10)
        
        D = np.abs(librosa.stft(signal))
        fig, ax = plt.subplots(nrows=2, sharex=True)
        librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max), 
                                 x_axis='time', y_axis='log', ax=ax[0])
        ax[0].set(title='Power spectrogram')
        ax[0].label_outer()
        ax[1].plot(times, o_env, label='Onset strength')
        ax[1].vlines(times[peaks], 0, o_env.max(), color='r', alpha=0.9, 
                     linestyle='--', label='Onsets')
        ax[1].legend()
        
        
        fig, ax = plt.subplots(figsize=(10,2))
        ax.plot(signal, label='signal')
        ax.vlines(librosa.frames_to_samples(peaks), signal.min(), signal.max(), 
                   color='r', alpha=0.9, linestyle='--', label='Onsets')
        ax.legend()
        
        
        
        S = np.abs(librosa.stft(y=signal))
        rms = librosa.feature.rms(S=S)
        onset_bt_rms = librosa.onset.onset_backtrack(peaks, rms[0])
        
        
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(nrows=2, sharex=True)
        librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
                                 y_axis='log', x_axis='time', ax=ax[0])
        ax[0].label_outer()
        ax[1].plot(times, rms[0], label='RMS')
        ax[1].vlines(times[onset_bt_rms], 0, rms.max(), label='Backtracked (RMS)', color='r')
        ax[1].legend()
        
        
        
        
        signal_duration = signal.shape[0] / sr
        # interval at which a new sensor data is received
        sensor_interval = signal_duration / acc_left_x.shape[0]
        sound2sensor = acc_left_x.shape[0] / signal.shape[0]
        sensor_peaks_left = librosa.frames_to_samples(onset_bt_rms) * sound2sensor
        
        fig, ax = plt.subplots(nrows=13, figsize=(10,10))
        ax[0].plot(signal, label='signal')
        ax[0].vlines(librosa.frames_to_samples(onset_bt_rms), signal.min(), signal.max(), 
                   color='r', alpha=0.9, linestyle='--', label='Onsets')
        ax[0].legend()
        ax[1].plot(acc_left_x, label='accelerometer')
        ax[1].vlines(sensor_peaks_left, acc_left_x.min(), acc_left_x.max(), 
                   color='r', alpha=0.9, linestyle='--', label='Onsets')
        ax[2].plot(acc_left_y, label='accelerometer')
        ax[2].vlines(sensor_peaks_left, acc_left_y.min(), acc_left_y.max(), 
                   color='r', alpha=0.9, linestyle='--', label='Onsets')
        ax[3].plot(acc_left_z, label='accelerometer')
        ax[3].vlines(sensor_peaks_left, acc_left_z.min(), acc_left_z.max(), 
                   color='r', alpha=0.9, linestyle='--', label='Onsets')
        
        ax[4].plot(gyro_left_x, label='accelerometer')
        ax[4].vlines(sensor_peaks_left, gyro_left_x.min(), gyro_left_x.max(), 
                   color='r', alpha=0.9, linestyle='--', label='Onsets')
        ax[5].plot(gyro_left_y, label='accelerometer')
        ax[5].vlines(sensor_peaks_left, gyro_left_y.min(), gyro_left_y.max(), 
                   color='r', alpha=0.9, linestyle='--', label='Onsets')
        ax[6].plot(gyro_left_z, label='accelerometer')
        ax[6].vlines(sensor_peaks_left, gyro_left_z.min(), gyro_left_z.max(), 
                   color='r', alpha=0.9, linestyle='--', label='Onsets')
        fig.suptitle('G4-vertical split', fontsize=15)
        
        '''
        signal_duration = signal.shape[0] / sr
        # interval at which a new sensor data is received
        sensor_interval = signal_duration / acc_right_x.shape[0]
        sound2sensor = acc_right_x.shape[0] / signal.shape[0]
        sensor_peaks_right = librosa.frames_to_samples(onset_bt_rms) * sound2sensor
        
        ax[7].plot(acc_right_x, label='accelerometer')
        ax[7].vlines(sensor_peaks_right, acc_right_x.min(), acc_right_x.max(), 
                   color='r', alpha=0.9, linestyle='--', label='Onsets')
        ax[8].plot(acc_right_y, label='accelerometer')
        ax[8].vlines(sensor_peaks_right, acc_right_y.min(), acc_right_y.max(), 
                   color='r', alpha=0.9, linestyle='--', label='Onsets')
        ax[9].plot(acc_right_z, label='accelerometer')
        ax[9].vlines(sensor_peaks_right, acc_right_z.min(), acc_right_z.max(), 
                   color='r', alpha=0.9, linestyle='--', label='Onsets')
        
        ax[10].plot(gyro_right_x, label='accelerometer')
        ax[10].vlines(sensor_peaks_right, gyro_right_x.min(), gyro_right_x.max(), 
                   color='r', alpha=0.9, linestyle='--', label='Onsets')
        ax[11].plot(gyro_right_y, label='accelerometer')
        ax[11].vlines(sensor_peaks_right, gyro_right_y.min(), gyro_right_y.max(), 
                   color='r', alpha=0.9, linestyle='--', label='Onsets')
        ax[12].plot(gyro_right_z, label='accelerometer')
        ax[12].vlines(sensor_peaks_right, gyro_right_z.min(), gyro_right_z.max(), 
                   color='r', alpha=0.9, linestyle='--', label='Onsets')
        fig.suptitle('G4-vertical split', fontsize=15)
        '''
        
        
        peak_samples = librosa.frames_to_samples(onset_bt_rms)
        segments = []
        previous_peak = 0
        for peak in peak_samples:
            segment = signal[previous_peak:peak]
            segments.append(segment)
            previous_peak = peak
        segments.append(signal[previous_peak:])
        audio_segments = segments[1:]
        #for segment in segments:
        #    fig, ax = plt.subplots(figsize=(10,2))
        #    ax.plot(segment)
        
        
        sensor_peaks_int = [int(x) for x in sensor_peaks_left]
        sensor_segments = []
        previous_peak = 0
        for peak in sensor_peaks_int:
            segment = acc_left_x[previous_peak:peak]
            sensor_segments.append(segment)
            previous_peak = peak
        sensor_segments.append(acc_left_x[previous_peak:])
        acc_left_x_segments = sensor_segments[1:]
        #for segment in sensor_segments:
        #    fig, ax = plt.subplots(figsize=(10,2))
        #    ax.plot(segment)
        
        sensor_segments = []
        previous_peak = 0
        for peak in sensor_peaks_int:
            segment = acc_left_y[previous_peak:peak]
            sensor_segments.append(segment)
            previous_peak = peak
        sensor_segments.append(acc_left_y[previous_peak:])
        acc_left_y_segments = sensor_segments[1:]
        
        sensor_segments = []
        previous_peak = 0
        for peak in sensor_peaks_int:
            segment = acc_left_z[previous_peak:peak]
            sensor_segments.append(segment)
            previous_peak = peak
        sensor_segments.append(acc_left_z[previous_peak:])
        acc_left_z_segments = sensor_segments[1:]
        
        sensor_segments = []
        previous_peak = 0
        for peak in sensor_peaks_int:
            segment = gyro_left_x[previous_peak:peak]
            sensor_segments.append(segment)
            previous_peak = peak
        sensor_segments.append(gyro_left_x[previous_peak:])
        gyro_left_x_segments = sensor_segments[1:]
        
        sensor_segments = []
        previous_peak = 0
        for peak in sensor_peaks_int:
            segment = gyro_left_y[previous_peak:peak]
            sensor_segments.append(segment)
            previous_peak = peak
        sensor_segments.append(gyro_left_y[previous_peak:])
        gyro_left_y_segments = sensor_segments[1:]
        
        sensor_segments = []
        previous_peak = 0
        for peak in sensor_peaks_int:
            segment = gyro_left_z[previous_peak:peak]
            sensor_segments.append(segment)
            previous_peak = peak
        sensor_segments.append(gyro_left_z[previous_peak:])
        gyro_left_z_segments = sensor_segments[1:]
        
        
        '''
        sensor_peaks_int = [int(x) for x in sensor_peaks_right]
        sensor_segments = []
        previous_peak = 0
        for peak in sensor_peaks_int:
            segment = acc_right_x[previous_peak:peak]
            sensor_segments.append(segment)
            previous_peak = peak
        sensor_segments.append(acc_right_x[previous_peak:])
        acc_right_x_segments = sensor_segments[1:]
        #for segment in sensor_segments:
        #    fig, ax = plt.subplots(figsize=(10,2))
        #    ax.plot(segment)
        
        sensor_segments = []
        previous_peak = 0
        for peak in sensor_peaks_int:
            segment = acc_right_y[previous_peak:peak]
            sensor_segments.append(segment)
            previous_peak = peak
        sensor_segments.append(acc_right_y[previous_peak:])
        acc_right_y_segments = sensor_segments[1:]
        
        sensor_segments = []
        previous_peak = 0
        for peak in sensor_peaks_int:
            segment = acc_right_z[previous_peak:peak]
            sensor_segments.append(segment)
            previous_peak = peak
        sensor_segments.append(acc_right_z[previous_peak:])
        acc_right_z_segments = sensor_segments[1:]
        
        sensor_segments = []
        previous_peak = 0
        for peak in sensor_peaks_int:
            segment = gyro_right_x[previous_peak:peak]
            sensor_segments.append(segment)
            previous_peak = peak
        sensor_segments.append(gyro_right_x[previous_peak:])
        gyro_right_x_segments = sensor_segments[1:]
        
        sensor_segments = []
        previous_peak = 0
        for peak in sensor_peaks_int:
            segment = gyro_right_y[previous_peak:peak]
            sensor_segments.append(segment)
            previous_peak = peak
        sensor_segments.append(gyro_right_y[previous_peak:])
        gyro_right_y_segments = sensor_segments[1:]
        
        sensor_segments = []
        previous_peak = 0
        for peak in sensor_peaks_int:
            segment = gyro_right_z[previous_peak:peak]
            sensor_segments.append(segment)
            previous_peak = peak
        sensor_segments.append(gyro_right_z[previous_peak:])
        gyro_right_z_segments = sensor_segments[1:]
        '''
                
        save_path = f'data-processed/{sample_name}'
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        
        for i in range(len(audio_segments)):
            out_f = f'{sample_name}-sound-{i}.wav'
            filepath = os.path.join(save_path, out_f)
            wavf.write(filepath, sr, audio_segments[i])
        
            sensor_filename = f'{sample_name}-sensors-left-{i}.npy'
            filepath = os.path.join(save_path, sensor_filename)
            sensor_data = []
            sensor_data.append(acc_left_x_segments[i])
            sensor_data.append(acc_left_y_segments[i])
            sensor_data.append(acc_left_z_segments[i])
            sensor_data.append(gyro_left_x_segments[i])
            sensor_data.append(gyro_left_y_segments[i])
            sensor_data.append(gyro_left_z_segments[i])
            min_length = np.array([data.shape for data in sensor_data]).min()
            sensor_data = np.array([data[:min_length] for data in sensor_data])
            np.save(filepath, sensor_data)
            
            '''
            sensor_filename = f'{sample_name}-sensors-right-{i}.npy'
            filepath = os.path.join(save_path, sensor_filename)
            sensor_data = []
            sensor_data.append(acc_right_x_segments[i])
            sensor_data.append(acc_right_y_segments[i])
            sensor_data.append(acc_right_z_segments[i])
            sensor_data.append(gyro_right_x_segments[i])
            sensor_data.append(gyro_right_y_segments[i])
            sensor_data.append(gyro_right_z_segments[i])
            min_length = np.array([data.shape for data in sensor_data]).min()
            sensor_data = np.array([data[:min_length] for data in sensor_data])
            np.save(filepath, sensor_data)
            '''
            
            