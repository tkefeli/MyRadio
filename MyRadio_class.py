#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
22/04/2023: * Tuner gain eklendi.
            * FM Demodülatör düzeltildi.
            * Eksenlerin fontları değiştirildi.

19/04/2023: * Sürekli LineCollection eklenmesi sonucu oluşan memeory-leak çözüldü.
            * waterfall için spektrumların ortalaması kullanıldı. Olması gereken buydu.
            * Spektrumda kırmızı marker ölçklendirme yapınca kayboluyordu, onu animasyon
              içerisine alınca memeory-leak durumu olmasın diye iki kez "collection"
              dizisinden veri silindi, sorun çözüldü.
            * waterfall grafiklerinin performansı deque kullanılarak iyileştirildi.
            * spektrumda alınan fft'nin daha düzgün sonuçlar vermesi için windowlama fonksiyonu eklendi.

17/04/2023: * AM Demodülatör kısmına kazanç eklendi.
            * Grafiklere kırmızı merkez çizgileri eklendi.
            * waterfall_buffer_size değişkeni eklendi (yerel buffer_size yerine)

15/04/2023: * Program çalıştırılabir (chmod +x) şekilde düzenlendi.
            * fft örnek sayısı da giriş argümanlarına eklendi.

14/04/2023: * Varsayılan değerler değiştirildi. (sampling-rate, bandwidth, dsize..)
            * Daha düzgün bir sonlanma sağlandı.
            * Radyo çalışırken önemli parametreler ekranda gösterildi.
            * RTL-SDR AGC için de programa bir opsiyon eklendi.
            
13/04/2023: * MyRadio.py programının class şeklinde düzenlenmesi.
"""

from rtlsdr import RtlSdr
from time import sleep
import argparse, multiprocessing
import numpy as np
import multiprocessing as mp
from collections import deque

parser = argparse.ArgumentParser(description='Simple radio program using RTL-SDR')
parser.add_argument('--frequency', dest='freq', type=float, help='RF tune frequency', required=True)
parser.add_argument('--sampling-rate', dest='srate', type=int, default=1024000, help='RTL-SDR sampling rate')
parser.add_argument('--bandwidth', dest='bandwidth', type=float, default=5000, help='Channel bandwidth')
parser.add_argument('--arate', dest='arate', type=int, default=48000, help='Sound device audio bandwidth')
parser.add_argument('--dsize', dest='dsize', type=int, default=51200, help='Data size')
parser.add_argument('--modulation', dest='mod', type=str, help='Modulation type', required=True)
parser.add_argument('--slevel', dest='slevel', type=float, help='Spectral noise blanking level', required=False, default=0)
parser.add_argument('--alevel', dest='alevel', type=float, help='Audio noise blanking level', required=False, default=0)
parser.add_argument('--spectrum', action='store_true', help='Enable spectrum view')
parser.add_argument('--waterfall', action='store_true', help="Enable waterfall view")
parser.add_argument('--noagc', action='store_true', help="Disable RTL-SDR AGC")
parser.add_argument('--nfft', dest='nfft', type=int, default=1024, help='The number of FFT Samples')
parser.add_argument('--volume', dest='volume', type=float, default=10, help='AM demodulator volume')
parser.add_argument('--gain', dest='gain', type=float, default=19.6, help='Tuner Gain')

args = parser.parse_args()

class MyRadio():

    def __init__(self):
        # Gerekli kuyruklar..
        self.qsdr = mp.Queue(1)
        self.qdemod = mp.Queue()        
        self.qvisual = mp.Queue(1)
        self.qstatus = mp.Queue(1)
        self.event = mp.Event()
                       
        # Giriş parametreleri burada..
        self.srate = args.srate
        self.arate = args.arate
        self.bw = args.bandwidth
        self.freq = args.freq
        self.dsize = args.dsize
        self.mod = args.mod
        self.slevel = args.slevel
        self.alevel = args.alevel
        self.spectrum = args.spectrum
        self.waterfall = args.waterfall
        self.noagc = args.noagc
        self.fft_size = args.nfft
        self.volume = args.volume
        self.gain = args.gain
        self.waterfall_buffer_size = 128 # Bu değer waterfall grafiği hazırlanırken self.fft_size değeri ile dinamik olarak değiştirilir.
        self.waterfall_data = deque(maxlen=self.waterfall_buffer_size)
        self.wfdata = np.zeros(self.fft_size)
        
        # Klavye kontrol parametreleri..
        self.vol_up, self.vol_down, self.prev_tune, self.next_tune = 0,0,0,0
        self.parameters = {"u":self.vol_up, "d":self.vol_down, "p":self.prev_tune, "n":self.next_tune}
        
        # Desimasyon ve demodülasyon parametreleri burda..
        self.decimation_factor = self.srate//self.arate # Bu resampling faktörü olacak. srate'den arate'e resample için.
        self.chunk_size = self.dsize//self.decimation_factor # Bu ses kartına tek seferde gidecek olan veri miktarı.
        self.buffer_size = self.dsize*3
        self.f_low = 10
        self.f_high = self.bw

    # Manual start burada başlıyor..
    def start(self):
        self.setupSdr()
        self.create_processes()
        self.start_processes()
        
    # RTL-SDR Burada koşullanır..    
    def setupSdr(self):
        self.sdr = RtlSdr()
        self.sdr.set_manual_gain_enabled(True) 
        self.sdr.set_gain(self.gain)
        self.sdr.set_agc_mode(not self.noagc)
        self.sdr.set_sample_rate(self.srate)
        self.sdr.set_center_freq(self.freq)
    
#     # Prosesler burada tanımlanıyor..
    def create_processes(self):
        self.streaming_process = mp.Process(target=self.Streaming)
        self.demodulator_process = mp.Process(target=self.Demodulation)
        self.audio_process = mp.Process(target=self.Sound)
        self.visual_process = mp.Process(target=self.Spectrum_and_waterfall)
        
    # Prosesler burada çalıştırılıyor..
    def start_processes(self):
        self.streaming_process.start()
        self.demodulator_process.start(),
        if self.spectrum: self.visual_process.start()
        sleep(1)
        self.audio_process.start()
    
    # Radio burada durdurulur..
    def stop(self):
        self.streaming_process.kill()
        self.demodulator_process.kill()
        self.audio_process.kill()
        if self.spectrum: self.visual_process.kill()
        self.sdr.close()
        
    # Spektral (frekans domeninde) filtre oluşturma..
    def create_filter_mask(self, f_low, f_high, lenght, rate):
        mask = np.zeros(lenght)
        low_f = int((f_low*lenght)//rate)
        high_f = int((f_high*lenght)//rate)
        mask[lenght//2-high_f:lenght//2+high_f] = 1
        mask[lenght//2-low_f:lenght//2+low_f] = 0
        return mask
    
    # Audio domeminde gürültü azaltma algoritması..
    def audio_spectral_subtraction(self, signal, level=0, block_dc=True):      
        level = level*np.ones(len(signal))
        specs = np.fft.fft(signal)
        if block_dc: specs[0] = 0
        amplitudes = np.abs(specs)
        phases = np.exp(1j*np.angle(specs))
        diff = np.maximum(amplitudes-level, 0)
        recovered = np.fft.ifft(diff*phases).real        
        return recovered

    # Spectrum ve waterfall verisi burada işleniyor..
    def Spectrum_and_waterfall(self):
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        from matplotlib.animation import FuncAnimation
        from matplotlib.colors import Normalize

        # frekans değerleri burada eksene ekleniyor.
        def bin2freq(x, pos):        
            step = self.srate/self.fft_size          
            return f'{((x-self.fft_size/2)*step+self.freq)*1e-6:1.2f} MHz'
        
        def dBs(x, pos):                       
            return f'{x:3.0f} dB'
        
        # Toplam çizim alanı ile ilgili yerler ve ilkleştirme.       
        plt.style.use('dark_background')
        plt.rcParams.update({
        'font.sans-serif': 'URW Gothic',
        'font.family': 'sans-serif'
                            })
        fig, (ax_spec, ax_wfall) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(10, 4))
        fig.set_tight_layout(True)
                
        # Spektrum ile ilgili yerler:
        norm = Normalize(-100, 0)
        ax_spec.set_ylim(-100, 0)
        ax_spec.set_xlim(0, self.fft_size)
        ax_spec.grid(linewidth=0.5, color="gray")
        ax_spec.xaxis.set_major_formatter(bin2freq)
        ax_spec.yaxis.set_major_formatter(dBs)
        ax_spec.vlines(self.fft_size//2, -100, 0, colors="red", linewidth=0.5) # Bu kırmızı renkli marker.
        
        # waterfall ile ilgili yerler
        waterfall_level_offset = -20 # Görünümün renkleri ile ilgili. 
        waterfall_buffer_size = self.waterfall_buffer_size*(self.fft_size//1024)  
        im_wfall = ax_wfall.imshow(np.random.randn(waterfall_buffer_size, self.fft_size), cmap="jet", animated=True, norm=norm)

        def generate_colored_spectrum_data(n, fft_size=self.fft_size, sampling_rate=self.srate):
            global data
            data = self.qvisual.get() if not self.qvisual.empty() else data # Veri burada alınır kuyruktan..            
            xdata = np.arange(fft_size)
            num_rows = int(np.floor(len(data)/fft_size))
            spectrogram = np.zeros((num_rows, fft_size))
            for i in range(num_rows):
                spectrogram[i,:] = 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(data[i*fft_size:(i+1)*fft_size]*np.hanning(fft_size))/fft_size))**2)            
            ydata = np.mean(spectrogram, axis=0) # Ortalamanın wfdata'ya verilmesi waterfall görüntüsünü iyileştiriyor. wfdata waterfall için gereken veriyi tutar.
            self.wfdata = ydata
            points = np.array([xdata, ydata]).T.reshape(-1, 1, 2)    
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(segments, cmap='jet', norm=norm, linewidths=0.5)
            lc.set_array(ydata)    
            ax_spec.add_collection(lc) # Çizim burada yapılıyor. Burada geriye de bir şey  dönüyor aslında ama kullanmıyoruz.
            ax_spec.vlines(fft_size//2, -100, 0, colors="red", linewidth=0.5) # Kırmızı marker burada ekleniyor.
            if n>10:
                ax_spec.collections[0].remove() # Bu eklenen LineCollection için.
                ax_spec.collections[0].remove() # Bu da eklenen vlines için. O da collections nesnesinin içine gidiyormuş.            
            return (lc,)

        def generate_waterfall(n, fft_size=self.fft_size, sampling_rate=self.srate, buffer_size=waterfall_buffer_size):            
            spec = self.wfdata + waterfall_level_offset # waterfall verileri wfdata değişkeninden gelir..            
            self.waterfall_data.appendleft(spec)
            temp = np.array(self.waterfall_data) # Bu aşağıdaki işi yapmak için gerekliydi.
            temp[:, fft_size//2] = 0 # waterfall'ın ortasındaki kırmızı merkez çizgisi. Aslında image'deki noktalar bütünü..
            im_wfall.set_array(temp)
            return (im_wfall,)
        
        interval = 1000*(self.dsize/self.srate)*(self.fft_size//1024)
        spec_ani = FuncAnimation(fig, generate_colored_spectrum_data, blit=True, interval=interval, cache_frame_data=False) # fig de dönebilir.
        if self.waterfall:
            wfall_ani = FuncAnimation(fig, generate_waterfall, blit=True, interval=interval, cache_frame_data=False) # plt.gcf() de dönebilir.
        plt.show()
   
    # RTL-SDR'den veri okunup kuyruğa yazılır. Eğer spectrum da gösterilecekse aynı veri başka
    # bir kuyruğa daha yazılır.
    def Streaming(self):
        if self.freq > 28000000:
            self.sdr.set_direct_sampling(0)
        else:
            self.sdr.set_direct_sampling(2)
        
        try:
            while True:
                data = self.sdr.read_samples(self.dsize)
                if self.qsdr.empty():
                    self.qsdr.put(data)
                if self.spectrum:
                    if self.qvisual.empty(): self.qvisual.put(data)
                
        except:
            self.sdr.close()
            print("streamingde sıkıntı çıktı !")

    # Demodülasyon kısmı buradan başlıyor..
    def Demodulation(self):        
        
    #     AM Demodülasyon kısmı...
        if self.mod == "AM":
            print("modulation type: AM")        
            mask = self.create_filter_mask(self.f_low, self.f_high, self.buffer_size, self.srate)
            data1 = self.qsdr.get()
            data2 = self.qsdr.get()
            data3 = self.qsdr.get() 
            data = np.array([data1, data2, data3])
            while True:
                if not self.qsdr.empty():
                    x = self.qsdr.get()*self.volume # Burada artık kazanç değeri dışarıdan verilebiliyor. 
                    data[-1] = x    
                    flat_data = data.flatten()                                     
                    z = np.fft.fftshift(np.fft.fft(flat_data))
                    z = z*mask
                    if self.slevel:
                        angles = np.angle(z)
                        amplitudes = abs(z)
                        z = np.maximum(amplitudes-self.slevel, 0)
                        z = z*np.exp(1j*angles)
                    filtered = np.fft.ifft(np.fft.ifftshift(z))
                    resampled = filtered[::self.decimation_factor]
                    demod = abs(resampled)
                    avg = np.mean(demod)
                    demod -= avg
                    if self.alevel:                    
                        demod = self.audio_spectral_subtraction(demod, level=self.alevel, block_dc=True)
                    demod = demod[self.chunk_size:2*self.chunk_size]
                    self.qdemod.put(demod.astype(np.float32))                
                    data[:-1] = data[1:]
                    
    #     FM Demodülasyon kısmı..            
        if self.mod == "FM":
            print("modulation type: FM")            
            mask1 = self.create_filter_mask(self.f_low, 80000, self.buffer_size, self.srate)
            mask2 = self.create_filter_mask(self.f_low, self.bw, self.buffer_size, self.srate)
            data1 = self.qsdr.get()
            data2 = self.qsdr.get()
            data3 = self.qsdr.get() 
            data = np.array([data1, data2, data3])
            initial = 0
            while True:
                x = self.qsdr.get()            
                data[-1] = x
                flat_data = data.flatten()                                     
                z1 = np.fft.fftshift(np.fft.fft(flat_data))
                z1 = z1*mask1 # FM Braadcast kanal için ilk filtreleme.. 
                filtered1 = np.fft.ifft(np.fft.ifftshift(z1)) # Reconstruction yapıldı. Şimdi demodüle edilip yeniden filtrelenecek..
                t = np.insert(filtered1, 0, initial)
                demod = 0.5 * np.angle(t[0:-1] * np.conj(t[1:])) # Burada zaman domeninde demodüle edildi.
                z2 = np.fft.fftshift(np.fft.fft(demod))
                z2 = z2*mask2 # Burada (L+R)'nin bulunduğu ilk 15 kHz'lik band filtrelenir.
                filtered2 = np.fft.ifft(np.fft.ifftshift(z2)) # Bilgi işareti reconstruct edildi.
                resampled = filtered2[0::self.decimation_factor]        
                resampled = resampled[self.chunk_size:2*self.chunk_size]
                data[:-1] = data[1:]
                initial = demod[-1]
                self.qdemod.put(resampled.real.astype(np.float32))
        
    # Ses kartına veri gönderen program burada..
    def Sound(self):
        import sounddevice as sd
        """ Anlaşılan o ki callback rutini, stream open edildikten sonra 
        periyodik olarak çağrılıyor. Ancak bu rutinin kesintisiz ses vermesi için
        veri aldığı yerde belirli bir miktar veri olmalıymış ki hardware ile sorun
        yaşamasın, o yüzden kuyruk biraz önden dolduruldu..
            Ayrıca callback rutini aşağıdaki gibi çağrılmalı (şeklen) ve frames
        parametresi, stream değişkeninin tanımlamasında blocksize olan değişken olarak 
        belirlenmektedir. frames sayısı (int) ses kartına gönderilen birim zamandaki
        veri miktarıdır.
        """
        def callback(outdata, frames, time, status):
            try:
                data = self.qdemod.get().astype(np.float32).reshape(self.chunk_size, 1)
            except:
                print("sorun")
            if len(data) < len(outdata):
                outdata[:len(data)] = data
                outdata[len(data):].fill(0)
                raise sd.CallbackStop
            else:
                outdata[:] = data
                
        stream = sd.OutputStream(device=sd.default.device, blocksize=self.chunk_size, 
                                  samplerate=self.arate,  channels=1,
                                  callback=callback, finished_callback=self.event.set)
        
        with stream: 
            self.event.wait()
        
if __name__ == "__main__":
    radio = MyRadio()
    radio.start()
    import os
    os.system("clear")
    banner = (f'{"   MY RADIO   ":*^80}\n'
              f'{"* Frequency":<20} {1e-6*radio.freq:>10.03f} MHz\n'
              f'{"* Sampling rate":<20} {1e-3*radio.srate:>10.0f} kSps\n'
              f'{"* Bandwidth":<20} {1e-3*radio.bw:>10.0f} kHz\n'
              f'{"* FFT Size":<20} {radio.fft_size:>10.0f} Sample\n'
              f'{"* RTL-SDR AGC":<20} {"ON" if not radio.noagc else "OFF":>10}\n'
              f'{"* Tuner Gain":<20} {radio.sdr.get_gain():>10.1f}\n'
              )
    print(banner)        
    print("press esc+enter for quit ..\n")
    while True:
        key = input()
        if key == '\x1b': # esc
            print("Esc key pressed, exiting...\n")            
            radio.stop()
            break
        elif key == "u":
            print("Volume up..")
        elif key == "d":
            print("Volume Down..")
        elif key == "n":
            print("Next station..")
        elif key == "p":
            print("Previous station..")