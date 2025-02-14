#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
27/09/2023: * "demodulate_fm_with_decimation_triple" fonksiyonundaki 0.5 katsayısı kaldırıldı.
23/09/2023: * "demodulate_fm_with_decimation_single" eklendi ve bu demodülasyonda 0.5 katsayısı kaldırıldı.

Created on Thu Jul 27 09:20:53 2023

@author: tansu
"""
import numpy as np

# Eğer basit "slicing" olmuyorsa bunu kullan.. 
def get_sample_points(lenght, slicing_factor):
    return np.array([int(np.rint(i*slicing_factor)) for i in range(lenght)])

# Burada fm sinyali tekli paketler halinde alınır ve slicing ile demodüle/resample edilir, bilgi işareti döner.
# pythran export demodulate_fm_with_decimation_single(complex128[], float[], float[], int, complex)
def demodulate_fm_with_decimation_single(FM, mask1, mask2, decimation_factor, initial):                                           
    z1 = np.fft.fft(FM)*mask1 # FM Braadcast kanal için ilk filtreleme..     
    filtered1 = np.fft.ifft(z1) # Reconstruction yapıldı. Şimdi demodüle edilip yeniden filtrelenecek..
    t = np.insert(filtered1, 0, initial)
    demod = np.angle(t[0:-1] * np.conj(t[1:])) # Burada zaman domeninde demodüle edildi.
    z2 = np.fft.fft(demod)*mask2 # Burada (L+R)'nin bulunduğu ilk 15 kHz'lik band filtrelenir.    
    filtered2 = np.fft.ifft(z2) # Bilgi işareti reconstruct edildi.
    resampled = filtered2[0::decimation_factor] 
    initial = filtered1[-1]
    return resampled, initial 

# Burada fm sinyali üçlü paketler halinde alınır ve slicing ile demodüle/resample edilir, bilgi işareti döner.
# pythran export demodulate_fm_with_decimation_triple(complex128[], complex128[:,:], float[], float[], int, int, int, complex)
def demodulate_fm_with_decimation_triple(FM, data, mask1, mask2, decimation_factor, chunk_size, dsize, initial):         
    data[-1] = FM
    flat_data = data.flatten()                                       
    z1 = np.fft.fft(flat_data)*mask1 # FM Braadcast kanal için ilk filtreleme..     
    filtered1 = np.fft.ifft(z1) # Reconstruction yapıldı. Şimdi demodüle edilip yeniden filtrelenecek..
    t = np.insert(filtered1, 0, initial)
    demod = np.angle(t[0:-1] * np.conj(t[1:])) # Burada zaman domeninde demodüle edildi.
    z2 = np.fft.fft(demod)*mask2 # Burada (L+R)'nin bulunduğu ilk 15 kHz'lik band filtrelenir.    
    filtered2 = np.fft.ifft(z2) # Bilgi işareti reconstruct edildi.
    resampled = filtered2[0::decimation_factor] 
    resampled = resampled[chunk_size:2*chunk_size]    
    data = np.roll(data, -dsize) # data "roll" edilirken önce "flatten", sonra roll, daha sonra reshape edilir.
    initial = filtered1[-1]
    return resampled, data, initial 

# Burasa resample, slicing pointler kullanılarak olur. Bilgi işareti geri döner.
# pythran export demodulate_fm_with_slicing_points_single(complex128[], int[], float[], float[], complex)
def demodulate_fm_with_slicing_points_single(FM, points, mask1, mask2, initial):         
    z1 = np.fft.fft(FM)*mask1 # FM Braadcast kanal için ilk filtreleme..     
    filtered1 = np.fft.ifft(z1) # Reconstruction yapıldı. Şimdi demodüle edilip yeniden filtrelenecek..
    t = np.insert(filtered1, 0, initial)
    demod = 0.5 * np.angle(t[0:-1] * np.conj(t[1:])) # Burada zaman domeninde demodüle edildi.    
    z2 = np.fft.fft(demod)*mask2 # Burada (L+R)'nin bulunduğu ilk 15 kHz'lik band filtrelenir.    
    filtered2 = np.fft.ifft(z2) # Bilgi işareti reconstruct edildi.
    resampled = filtered2[points]
    initial = filtered1[-1]            
    return resampled, initial

# Burada resample edilmeden yapılan demodülasyon var. Tekil paket kullanılarak.
# pythran export demodulate_fm_without_resample_single(complex128[], float[], float[], complex)
def demodulate_fm_without_resample_single(FM, mask1, mask2, initial):         
    z1 = np.fft.fft(FM)*mask1 # FM Braadcast kanal için ilk filtreleme..     
    filtered1 = np.fft.ifft(z1) # Reconstruction yapıldı. Şimdi demodüle edilip yeniden filtrelenecek..
    t = np.insert(filtered1, 0, initial)
    demod = 0.5 * np.angle(t[0:-1] * np.conj(t[1:])) # Burada zaman domeninde demodüle edildi.    
    z2 = np.fft.fft(demod)*mask2 # Burada (L+R)'nin bulunduğu ilk 15 kHz'lik band filtrelenir.    
    filtered2 = np.fft.ifft(z2) # Bilgi işareti reconstruct edildi.    
    initial = filtered1[-1]            
    return filtered2, initial 

# Normal demodulate_fm fonksiyonunun, 3'lü paketler kullanılarak demodüle edilmesi.
# pythran export demodulate_fm_without_resample_triple(complex128[], complex128[:,:], float[], float[], complex)
def demodulate_fm_without_resample_triple(FM, data, mask1, mask2, initial):         
    data[-1] = FM
    flat_data = data.flatten()                                       
    z1 = np.fft.fft(flat_data)*mask1 # FM Braadcast kanal için ilk filtreleme..     
    filtered1 = np.fft.ifft(z1) # Reconstruction yapıldı. Şimdi demodüle edilip yeniden filtrelenecek..
    t = np.insert(filtered1, 0, initial)
    demod = 0.5 * np.angle(t[0:-1] * np.conj(t[1:])) # Burada zaman domeninde demodüle edildi.
    z2 = np.fft.fft(demod)*mask2 # Burada (L+R)'nin bulunduğu ilk 15 kHz'lik band filtrelenir.    
    filtered2 = np.fft.ifft(z2) # Bilgi işareti reconstruct edildi.    
    data[:-1] = data[1:] # roll yerine bunu yaptım. dsize parametresinden kurtulduk.
#     data = np.roll(data, -dsize) # data "roll" edilirken önce "flatten", sonra roll, daha sonra reshape edilir.
    initial = filtered1[-1]
    return filtered2, data, initial 