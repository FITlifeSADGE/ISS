import numpy as np
import math
from scipy.io import wavfile
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.signal import spectrogram, lfilter, tf2zpk, freqz

def print_graph(fs, data, num):
    t = np.arange(data.size) / fs

    plt.figure(figsize=(6,3))
    plt.plot(t, data)

# plt.gca() vraci handle na aktualni Axes objekt, 
# ktery nam umozni kontrolovat ruzne vlastnosti aktualniho grafu
    plt.gca().set_xlabel('$t[s]$')
    #plt.gca().set_ylabel('$frekvence[Hz]$')
    plt.gca().set_title('Úloha '+num)

    plt.tight_layout()
    plt.show()

#Základy - Určí délku ve vzorcích a v sekundách, určí maximální a minimální hodnotu, zobrazí je s časovou osou v sekundách
def first_task(fs, data):
    #print("Vzorkovací frekvence:", fs) #Vzorkovací frekvence
    print("Délka v sekundách:", data.size/fs) #Délka v sekundách
    print("Minimální hodnota:", min(data)) #Minimální hodnota
    print("Maximální hodnota:", max(data)) #Maximální hodnota
    print("Počet vzorků:", data.size) #Počet vzorků

    print_graph(fs, data, '1')

#Předzpracování a rámce - Ustřední a normalizuje do dynamického rozsahu -1 až 1
def second_task(fs, data):
    avg = 0
    i = 0
    while i!=len(data):
        avg += data[i]
        i += 1

    avg = avg/data.size
    data = data - avg #ustředněné data
    data = data/max(abs(data)) #normalizované data

    i = 0
    cnt = 0
    array1 = np.array([])
    matrix1 = np.array([])

    while i!=len(data):
        if cnt <= 1023: #načte 1024 vzorků
            array1 = np.insert(array1,cnt,data[i]) #uloží vzorek do pole
            if cnt == 1023:
                matrix1 = np.append(matrix1,array1) #uloží pole do nového pole
                array1 = np.array([])
                cnt = -1
                i -= 512
        
        cnt += 1
        i += 1  
    matrix1 = np.reshape(matrix1, (-1, 1024)) #rámce po 1024 vzorcích s přkrytím 512 vzorků, uloženo do matice po sloupcích

    t = np.arange(matrix1[24].size) / fs
    t = t + ((data.size/fs / 130)*24)

    plt.figure(figsize=(6,3))
    plt.plot(t, matrix1[24])

    # plt.gca() vraci handle na aktualni Axes objekt, 
    # ktery nam umozni kontrolovat ruzne vlastnosti aktualniho grafu
    plt.gca().set_xlabel('$t[s]$')
    plt.gca().set_title('Úloha '+'2')

    plt.tight_layout()
    plt.show()

    #print_graph(fs, matrix1[24], '2')
    return matrix1[24]

def gen_basis(n):
    N = 1024
    e = math.e
    pi = math.pi
    return np.array([e**(-1j*2*pi*n*k/N) for k in np.arange(N)])

def dft_with_basis(_x, basis):
    N = 1024
    x = np.pad(_x, (0, N-len(_x)))
    return basis @ x.T


def third_task(fs, data):
    matr = np.array([])
    i = 0
    while i != len(data):
        basis = gen_basis(i)
        buff = dft_with_basis(data, basis)
        matr = np.append(matr, buff)
        i += 1
    #matr = np.fft.fft(data)
    matrix2 = np.array([])
    for i in range(len(matr)//2):
        matrix2 = np.insert(matrix2, i, matr[i])

    plt.figure(figsize=(10, 5))
    x = np.arange(matrix2.size)/1024*fs
    y = np.abs(matrix2)
    plt.title('4.3 - DFT - vlastní implementace')
    plt.gca().set_xlabel('Frekvence [Hz]')
    plt.plot(x, y)
    plt.show()
    #zobrazení knihovní implementace
    matr2 = np.fft.fft(data)
    matrix3 = np.array([])
    for i in range(len(matr2)//2):
        matrix3 = np.insert(matrix3, i, matr2[i])

    plt.figure(figsize=(10, 5))
    x = np.arange(matrix3.size)/1024*fs
    y = np.abs(matrix3)
    plt.title('4.3 - DFT - vestavěná funkce')
    plt.gca().set_xlabel('Frekvence [Hz]')
    plt.plot(x, y)
    plt.show()

    bool = np.allclose(matr, matr2, rtol=1e-05, atol=1e-08)
    print(bool)

def fourth_task(fs, data):
    #normalizace a ustřednění
    avg = 0
    i = 0
    while i!=len(data):
        avg += data[i]
        i += 1

    avg = avg/data.size
    data = data - avg #ustředněné data
    data = data/max(abs(data)) #normalizované data

    f, t, sgr = spectrogram(data, fs/2, nperseg=1024, noverlap=512)
    matrix1 = 10 * np.log10(sgr)

    # prevod na PSD
    # (ve spektrogramu se obcas objevuji nuly, ktere se nelibi logaritmu, proto +1e-20)
    plt.figure(figsize=(9,3))
    plt.imshow(matrix1, origin="lower",aspect="auto",extent=[0,matrix1.size/fs,0,fs/2])
    plt.title('4.4 - Spectrogram')
    plt.gca().set_xlabel('Čas [s]')
    plt.gca().set_ylabel('Frekvence [Hz]')
    cbar = plt.colorbar()
    cbar.set_label('Spektralní hustota výkonu [dB]', rotation=270, labelpad=15)

    plt.tight_layout()
    plt.show()

def fifth_task():
    real_values = [775, 1555, 2320, 3090]
    expected = [775, 775*2, 775*3, 775*4]
    print(real_values, expected)

def sixth_task(fs, data):
    array = np.array([])
    for i in range(len(data)):
        array = np.append(array, i/fs)

    output_cos1 = np.cos(2 * np.pi * 775 * array)
    output_cos2 = np.cos(2 * np.pi * 1550 * array)
    output_cos3 = np.cos(2 * np.pi * 2325 * array)
    output_cos4 = np.cos(2 * np.pi * 3100 * array)

    out = np.array([])
    out = output_cos1+output_cos2+output_cos3+output_cos4

    f, t, sgr = spectrogram(out, fs, nperseg = 1024)
    sgr = 10 * np.log10(sgr)
    plt.figure(figsize=(9,3))
    plt.imshow(sgr, origin="lower",aspect="auto",extent=[0,out.size/fs,0,fs/2])
    plt.title('4.6 - Spectrogram')
    plt.gca().set_xlabel('Čas [s]')
    plt.gca().set_ylabel('Frekvence [Hz]')
    cbar = plt.colorbar()
    cbar.set_label('Spektralní hustota výkonu [dB]', rotation=270, labelpad=15)

    plt.tight_layout()
    plt.show()

    avg = 0
    i = 0
    while i!=len(out):
        avg += out[i]
        i += 1

    avg = avg/out.size
    out = out - avg #ustředněné data
    out = out/max(abs(out)) #normalizované data

    #out = out.astype(np.int16)
    wavfile.write("../audio/4cos.wav", 16000, (out * np.iinfo(np.int16).max).astype(np.int16))

def seventh_task(fs):
    freq = [775, 1550, 2325, 3100]

    compl = np.array([])
    for i in range(0, 4):
        calc = 2*math.pi*(freq[i]/fs)
        calc = math.e**(1j*calc)
        compl = np.append(compl, calc)
    compl_conj = np.array([])
    compl_conj = np.conjugate(compl)
    final = np.array([])
    final = np.append(compl, compl_conj)
    final1 = np.poly(final)

    N_imp = 9
    imp = [1, *np.zeros(N_imp-1)] # jednotkovy impuls
    h = lfilter(final1, [1], imp)

    plt.figure(figsize=(5,3))
    plt.stem(np.arange(N_imp), h, basefmt=' ')
    plt.gca().set_xlabel('$n$')
    plt.gca().set_title('Impulsní odezva $h[n]$')

    plt.grid(alpha=0.5, linestyle='--')

    plt.tight_layout()
    plt.show()

    print('koeficienty filtru jsou:', final1)
    return final1

def eighth_task(fs, data, filter):

    z, p, k = tf2zpk(filter, [1])

    plt.figure(figsize=(4,3.5))

    # jednotkova kruznice
    ang = np.linspace(0, 2*np.pi,100)
    plt.plot(np.cos(ang), np.sin(ang))

    # nuly, poly
    plt.scatter(np.real(z), np.imag(z), marker='o', facecolors='none', edgecolors='r', label='nuly')
    plt.scatter(np.real(p), np.imag(p), marker='x', color='g', label='póly')

    plt.gca().set_xlabel('Realná složka $\mathbb{R}\{$z$\}$')
    plt.gca().set_ylabel('Imaginarní složka $\mathbb{I}\{$z$\}$')

    plt.grid(alpha=0.5, linestyle='--')
    plt.legend(loc='upper left')

    plt.tight_layout()
    plt.show()

def ninth_task(fs, filter):
    w, H = freqz(filter, [1])

    plt.figure(figsize=(5,3))

    plt.plot(w / 2 / np.pi * fs, np.abs(H))
    plt.gca().set_xlabel('Frekvence [Hz]')
    plt.gca().set_title('Modul frekvenční charakteristiky $|H(e^{j\omega})|$')

    plt.grid(alpha=0.5, linestyle='--')

    plt.tight_layout()
    plt.show()

def tenth_task(fs, data, filter):
    avg = 0
    i = 0
    while i!=len(data):
        avg += data[i]
        i += 1

    avg = avg/data.size
    data = data - avg #ustředněné data
    data = data/max(abs(data)) #normalizované data

    final = lfilter(filter, [1], data)

    f, t, sgr = spectrogram(final, fs, nperseg = 1024)
    sgr = 10 * np.log10(sgr)
    plt.figure(figsize=(9,3))
    plt.imshow(sgr, origin="lower",aspect="auto",extent=[0,final.size/fs,0,fs/2])
    plt.title('4.10 - Vyfiltrovaný signál')
    plt.gca().set_xlabel('Čas [s]')
    plt.gca().set_ylabel('Frekvence [Hz]')
    cbar = plt.colorbar()
    cbar.set_label('Spektralní hustota výkonu [dB]', rotation=270, labelpad=15)

    plt.tight_layout()
    plt.show()


    avg = 0
    i = 0
    while i!=len(final):
        avg += final[i]
        i += 1

    avg = avg/final.size
    final = final - avg #ustředněné data
    final = final/max(abs(final)) #normalizované data

    print_graph(fs, final, '10')
    #final = final.astype(np.int16)
    wavfile.write("../audio/clean_z.wav", 16000, (final * np.iinfo(np.int16).max).astype(np.int16))


fs, data = wavfile.read("../audio/xkapra00.wav") #Načtení vstupního signálu
first_task(fs, data)
matrix = second_task(fs, data)
third_task(fs, matrix)
fourth_task(fs, data)
fifth_task()
sixth_task(fs, data)
filter = seventh_task(fs)
eighth_task(fs, data, filter)
ninth_task(fs, filter)
tenth_task(fs, data, filter)






