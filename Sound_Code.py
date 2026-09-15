import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import butter, lfilter


#Hombres
#audio_file = "C:\\Users\\aleja\\OneDrive\\Documents\\ELECTIVA 2\\PROJECT2\\Hombre.ogg" 
#audio_file = "C:\\Users\\aleja\\OneDrive\\Documents\\ELECTIVA 2\\PROJECT2\\HombreA.ogg" 

#Niños  
#audio_file = "C:\\Users\\aleja\\OneDrive\\Documents\\ELECTIVA 2\\PROJECT2\\isaac.ogg" 
#audio_file = "C:\\Users\\aleja\\OneDrive\\Documents\\ELECTIVA 2\\PROJECT2\\samara.ogg"

#Mujeres
#audio_file = "C:\\Users\\aleja\\OneDrive\\Documents\\ELECTIVA 2\\PROJECT2\\Mujer1.ogg"
audio_file = "C:\Users\samue\Desktop\Clases UCO\Electiva\Project-2-Sound\Audio_Files\Mujer2.ogg"


y, sr = librosa.load(audio_file, sr=None)
duration = librosa.get_duration(y=y, sr=sr)
n_channels = 1 if y.ndim == 1 else y.shape[1]



non_silent_indices = librosa.effects.split(y, top_db=20)

if len(non_silent_indices) > 0:
    filtered_audio = np.concatenate([y[start:end] for start, end in non_silent_indices])
else:
    filtered_audio = y  # Fallback si no se detectan segmentos no silenciosos

print(f"sr del audio original: {sr} Hz")
print(f"Nyquist: {sr/2} Hz")

desired_sample_rate = 8000
nyquist_freq = desired_sample_rate / 2
print(f"--- Justificación de muestreo (Nyquist) ---")
print(f"Frecuencia de muestreo elegida: {desired_sample_rate} Hz")
print(f"Frecuencia máxima capturable (Nyquist): {nyquist_freq} Hz")
print(f"Suficiente para voz humana (energía relevante hasta ~4 kHz)\n")


resampled_audio = librosa.resample(filtered_audio, orig_sr=sr, target_sr=desired_sample_rate)



def butter_filter(data, cutoff, fs, btype, order=5):
    nyq = 0.5 * fs #is the Nyquist frequency of the signal you're about to filter.
    #needs the cutoff frequencies between 0 and 1, where 1 = Nyquist), so you divide the cutoff(s) by ny
    #If cutoff is a list (bandpass case, two frequencies), it normalizes each one; if it's a single number
    if isinstance(cutoff, (list, tuple)):
        normal_cutoff = [c / nyq for c in cutoff] 
    else: 
        normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype=btype, analog=False)
    return lfilter(b, a, data)

lowpass  = butter_filter(y, 4000, sr, btype="low") 
bandpass = butter_filter(y, [30, 3400], sr, btype="band")


plt.figure(figsize=(10, 4))
t = np.arange(len(y)) / sr
plt.plot(t, y, alpha=0.6, label="Original")
plt.plot(t, lowpass, alpha=0.8, label="Filtrada (lowpass 4kHz)")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")
plt.title("Efecto del filtro pasabajas sobre la señal")
plt.legend()
plt.tight_layout()
plt.show()


#denoising


D = librosa.stft(y)
S = np.abs(D)

n_fft = int(0.5 * sr)
noise_profile = np.mean(np.abs(D[:, :int(n_fft / 512)]), axis=1, keepdims=True)
S_denoised = np.maximum(S - noise_profile, 0)



bit_depth = 8

normalized_audio = resampled_audio / np.max(np.abs(resampled_audio))
max_amplitude = 2 ** (bit_depth - 1) - 1
quantized_audio = np.round(normalized_audio * max_amplitude).astype(np.int16)

# Visualizar señal normalizada (continua) vs cuantizada (escalonada)
plt.figure(figsize=(10, 4))
n_samples_plot = 200  # solo un tramo corto para que se vea el escalonado
t_q = np.arange(n_samples_plot) / desired_sample_rate
plt.step(t_q, quantized_audio[:n_samples_plot], where="mid", label=f"Cuantizada ({bit_depth} bits)")
plt.plot(t_q, normalized_audio[:n_samples_plot] * max_amplitude, alpha=0.5, label="Normalizada (continua)")
plt.xlabel("Tiempo (s)")
plt.ylabel("Nivel cuantizado")
plt.title("Cuantización: señal continua vs niveles discretos")
plt.legend()
plt.tight_layout()
plt.show()



def to_pcm_binary(samples, bit_depth):
    return [format(int(s) & (2**bit_depth - 1), f'0{bit_depth}b') for s in samples]

pcm_bits = to_pcm_binary(quantized_audio, bit_depth)
print("--- Coding (PCM binario) ---")
print(f"Primeras 5 muestras cuantizadas: {quantized_audio[:5]}")
print(f"Primeras 5 muestras codificadas en binario ({bit_depth} bits): {pcm_bits[:5]}\n")



plt.figure(figsize=(10, 4))
t_orig = np.arange(len(y)) / sr
t_proc = np.arange(len(filtered_audio)) / sr
plt.plot(t_orig, y, alpha=0.6, label="Original")
plt.plot(t_proc, filtered_audio, alpha=0.6, label="Procesada (silencios removidos)")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")
plt.title("Waveform: señal original vs procesada")
plt.legend()
plt.tight_layout()
plt.show()



n = len(filtered_audio)
T = 1.0 / sr
yf = fft(filtered_audio)
xf = fftfreq(n, T)[:n // 2]
amplitude = 2.0 / n * np.abs(yf[:n // 2])

f0, voiced_flag, voiced_prob = librosa.pyin(filtered_audio, fmin=50, fmax=400, sr=sr)
f0_valid = f0[~np.isnan(f0)]

# --- 7) Bug corregido: dejar explícito qué método se usó ---
if len(f0_valid) > 0:
    dominant_frequency = np.median(f0_valid)
    metodo_usado = "mediana de F0 (pyin)"
else:
    dominant_frequency = xf[np.argmax(amplitude)]
    metodo_usado = "pico de amplitud FFT (fallback, pyin no detectó voz)"

print(f"--- Frecuencia dominante ---")
print(f"Método usado: {metodo_usado}")
print(f"Frecuencia dominante: {dominant_frequency:.2f} Hz\n")

# --- 4) Rango de frecuencia numérico de la voz ---
if len(f0_valid) > 0:
    f0_min = np.min(f0_valid)
    f0_max = np.max(f0_valid)
    print(f"--- Rango de frecuencia de la voz ---")
    print(f"Rango de frecuencia: {f0_min:.2f} Hz - {f0_max:.2f} Hz\n")
else:
    f0_min, f0_max = None, None
    print("No se detectaron segmentos con F0 válido.\n")

# Clasificación aproximada del hablante
if dominant_frequency < 165:
    speaker_type = "Adult man"
elif dominant_frequency < 255:
    speaker_type = "Adult woman"
else:
    speaker_type = "Child"

print(f"The speaker is classified as: {speaker_type}")

eps = 1e-12
power_watts = (amplitude / np.sqrt(2)) ** 2 / 1.0
power_mw = power_watts * 1000
amp_dbm = 10 * np.log10(power_mw + eps)

mask = (xf >= 0) & (xf <= 300)
xf_plot = xf[mask]
amp_dbm_plot = amp_dbm[mask]

plt.figure(figsize=(10, 6))
plt.plot(xf_plot, amp_dbm_plot, label="Frequency Spectrum (dBm)")

if 0 <= dominant_frequency <= 300:
    plt.axvline(x=dominant_frequency, linestyle='--', color='r',
                label=f"Dominant: {dominant_frequency:.2f} Hz")
    if f0_min is not None:
        plt.axvspan(f0_min, f0_max, alpha=0.15, color='orange', label="Rango F0")

plt.xlabel("Frequency (Hz)")
plt.ylabel("Power (dBm)")
plt.title("Frequency Spectrum")
plt.plot([], [], ' ', label=f"The speaker is classified as: {speaker_type}")
plt.legend()
plt.tight_layout()
plt.show()