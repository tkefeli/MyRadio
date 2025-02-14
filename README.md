# MyRadio
The RTL-SDR command line radio application for python3. 
This the simple radio program written in python (version 3.xx) to listen to radio broadcasts HF band and FM band. 
The program uses a RTL-SDR dongle (at this time only RTL-SDR) to capture signals that come
from a passive loop antenna (which is sold from RTL-SDR site). There is no GUI right now, however the important parameters can be 
adjusted using command line parameters. 

The informative help mesage can be seen with --help keyword, like this:

python3 MyRadio_class.py --help

usage: MyRadio_class.py [-h] --frequency FREQ [--sampling-rate SRATE]
                        [--bandwidth BANDWIDTH] [--arate ARATE]
                        [--dsize DSIZE] --modulation MOD [--slevel SLEVEL]
                        [--alevel ALEVEL] [--spectrum] [--waterfall] [--noagc]
                        [--nfft NFFT] [--volume VOLUME] [--gain GAIN]

Simple radio program using RTL-SDR

options:

  -h, --help            show this help message and exit
  
  --frequency FREQ      RF tune frequency, in Hz, it must be specified.
  
  --sampling-rate SRATE
                        RTL-SDR sampling rate, default value: 1024000
                        
  --bandwidth BANDWIDTH
                        Channel bandwidth, the demodulation bandwidth, default value: 5000 Hz.
                        
  --arate ARATE         The sample rate of Sound device of system, default: 48000 
  
  --dsize DSIZE         The lenght of data read from RTL-SDR device, default: 51200
  
  --modulation MOD      Modulation type, actually FM or AM, it must be specified.
  
  --slevel SLEVEL       Spectral noise blanking level, default: 0
  
  --alevel ALEVEL       Audio noise blanking level, default: 0
  
  --spectrum            Enable spectrum view, default: disabled
  
  --waterfall           Enable waterfall view, default: disabled
  
  --noagc               Disable RTL-SDR AGC, default: disabled
  
  --nfft NFFT           The number of FFT Samples, default: 1024
  
  --volume VOLUME       AM demodulator volume, only for AM, default: 20
  
  --gain GAIN           The RF Tuner Gain, not RTL-SDR, default: 19.6 

Only two parameters needed to specify in order to run the radio program. The center frequency of broadcast signal and the modulation 
type of signal. The live spectrum and waterfall graphics can be seen using --spectrum and --waterfall keywords. The graphics are constructed
using matplotlib library. 

According to power level of broadcasting signal, two noise reduction/blanking scheme are applied on the demodulated signal. The first one is 
"spectral noise blanking", that is applied on the datas that captured from RTL-SDR before demodulating process, which can be selected the 
blanking level using --slevel keyword. The second one is "audio noise blanking" scheme that is applied after the modulation, using the --alevel 
keyword. 

The radio program uses multiprocessing, matplotlib, pyrtlsdr, argparse, numpy, time, collections and sounddevice python modules to run. 
This program has been written under Linux operating system (Mint, Ubuntu) and runs using multiprocessing mechanism, but under windows operating
system, The "threading" modules can be used instead of "processing" if some error messages arise that related with processing mechanism.

Not: I've forgotten to include the library file required for the FM demodulation, beginning with "demodulate_rm_ran". This file is built using
"pythran". Simply download the demodulate_fm_ran.py file and compile the file with the command below:

pythran demodulate_fm_ran

This command generates the required library. The name of the genarated file begins with "demodulate_rm_ran" and a long string behind of its name.
Do not change the file name including the extension, simply copy the generated file to the same folder with the MyRadio_class.py file.

In order to run MyRadio_class.py file, you need numpy, scipy, matplotlib, pyrtlsdr (for python), librtlsdr (pyrtlsdr library calls this library) and
pythran packages.
