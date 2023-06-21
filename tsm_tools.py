"""Time Series Modulation

Code is from https://github.com/HMC-MIR/PianoConcertoAccompaniment/blob/main/tsm_tools.ipynb and https://www.mdpi.com/128016

"""

import numpy as np
import librosa as lb
from scipy.signal import medfilt

def estimateIF_var(S, sr, timestamps):

    '''
    Estimates the instantaneous frequencies in an STFT-like matrix when the analysis frames
    are not evenly spaced.
    
    Inputs
    S: the STFT-like matrix, should only contain the lower half of the frequency bins
    sr: sampling rate
    timestamps: timestamps corresponding to each STFT column (in sec)
    
    Returns a matrix containing the estimated instantaneous frequency at each time-frequency bin.
    This matrix should contain one less column than S.
    
    '''
    assert S.shape[1] == len(timestamps)

    fft_size = (S.shape[0] - 1) * 2
    w_nom = np.arange(S.shape[0]) * sr / fft_size * 2 * np.pi
    w_nom = w_nom.reshape((-1,1))    
    unwrapped = np.angle(S[:,1:]) - np.angle(S[:,0:-1]) - w_nom * (timestamps[1:] - timestamps[0:-1]).reshape((1,-1))
    wrapped = (unwrapped + np.pi) % (2 * np.pi) - np.pi
    w_if = w_nom + wrapped / (timestamps[1:] - timestamps[0:-1]).reshape((1,-1))
    return w_if


def harmonic_percussive_separation(x, sr=22050, fft_size = 2048, hop_length=512, lh=6, lp=6):
    '''Performs harmonic-percussive separation on a given audio sample.
    
    Args
        x: the input audio waveform
        sr: sample rate of the audio data
        fft_size: specifies the FFT size to use in computing an STFT
        hop_length: specifies the hop length in samples to use in computing an STFT
        lh: the harmonic spectrotemporal median filter will be of size (1, 2*lh+1)
        lp: the percussive spectrotemporal median filter will be of size (2*lp+1, 1)
    
    Returns:
        xh: audio waveform of the estimated harmonic component
        xp: audio waveform of the estimated percussive component
        Xh: an STFT of the input signal with percussive components masked out
        Xp: an STFT of the input signal with harmonic components masked out
    '''
    
    window = hann_window(fft_size)
    X = lb.core.stft(x, n_fft=fft_size, hop_length=512, window=window, center=False)
    Y = np.abs(X)
    Yh = medfilt(Y, (1, 2*lh+1))
    Yp = medfilt(Y, (2*lp+1, 1))
    Mh = (Yh > Yp)
    Mp = np.logical_not(Mh)
    Xh = X * Mh
    Xp = X * Mp
    xh = invert_stft(Xh, hop_length, window)
    xp = invert_stft(Xp, hop_length, window)
    
    return xh, xp, Xh, Xp

def hann_window(L):
    '''Returns a Hann window of a specified length.
    
    Args:
        L: length of window to return
    '''
    w = .5 * (1 - np.cos(2*np.pi * np.arange(L)/ L))
    return w

def invert_stft(S, hop_length, window):
    '''Reconstruct a signal from a modified STFT matrix.
    
    Args:
        S: modified STFT matrix
        hop_length: the synthesis hop size in samples
        window: an array specifying the window used for FFT analysis
    
    Returns:
        A time-domain signal y whose STFT is closest to S in squared error distance.
    '''
    
    L = len(window)
    
    # construct full stft matrix
    fft_size = (S.shape[0] - 1) * 2
    Sfull = np.zeros((fft_size, S.shape[1]), dtype=np.complex64)
    Sfull[0:S.shape[0],:] = S
    Sfull[S.shape[0]:,:] = np.conj(np.flipud(S[1:fft_size//2,:]))
    
    # compute inverse FFTs
    frames = np.zeros_like(Sfull)
    for i in range(frames.shape[1]):
        frames[:,i] = np.fft.ifft(Sfull[:,i])
    frames = np.real(frames) # remove imaginary components due to numerical roundoff
    
    # synthesis frames
    num = window.reshape((-1,1))
    den = calc_sum_squared_window(window, hop_length)
    #den = np.square(window) + np.square(np.roll(window, hop_length))
    frames = frames * window.reshape((-1,1)) / den.reshape((-1,1))
    #frames = frames * window.reshape((-1,1))
    
    # reconstruction
    y = np.zeros(hop_length*(frames.shape[1]-1) + L)
    for i in range(frames.shape[1]):
        offset = i * hop_length
        y[offset:offset+L] += frames[:,i]
    
    return y

def calc_sum_squared_window(window, hop_length):
    '''Calculates the denominator term for computing synthesis frames.
    
    Inputs:
        window: array specifying the window used in FFT analysis
        hop_length: the synthesis hop size in samples
    
    Returns:
        An array specifying the normalization factor.
    '''
    assert (len(window) % hop_length == 0), "Hop length does not divide the window evenly."
    
    numShifts = len(window) // hop_length
    den = np.zeros_like(window)
    for i in range(numShifts):
        den += np.roll(np.square(window), i*hop_length)
        
    return den

def tsm_phase_vocoder(x, alpha = 1.0, L = 2048, sr = 22050):
    '''Time stretches the input signal using the phase vocoder method.
    
    Uses a synthesis hop size that is one-fourth the value of L.  Note that this implementation allows for a non-integer analysis hop size
    (in samples), which ensures that the time-scale modification factor is exactly as specified.
    
    Args:
        x: the input signal
        alpha: the time stretch factor, which is defined as the ratio of the synthesis hop size to the analysis hop size
        L: the length of each analysis frame in samples
        sr: sampling rate
    
    Returns:
        the time-stretched signal y.
    '''
    assert(L % 4 == 0), "Frame length must be divisible by four."
    Hs = L // 4
    
    # compute STFT
    Ha = Hs/alpha # allow non-integer values
    window = hann_window(L)
    X, analysis_frame_offsets = my_stft(x, L, Ha, window) # custom implementation to handle non-integer hop size
    
    # compute modified STFT
    w_if = estimateIF_var(X, sr, analysis_frame_offsets/sr) # custom implementation to handle non-constant frame locations
    phase_mod = np.zeros(X.shape)
    phase_mod[:,0] = np.angle(X[:,0]) 
    for i in range(1, phase_mod.shape[1]):
        phase_mod[:,i] = phase_mod[:,i-1] + w_if[:,i-1] * Hs / sr
    Xmod = np.abs(X) * np.exp(1j * phase_mod)
    
    # signal reconstruction
    y = invert_stft(Xmod, Hs, window)
    #y = lb.core.istft(Xmod, hop_length=Hs, center=False)
    
    return y

def my_stft(x, N, hop_length, window):
    '''A custom implementation of the STFT that allows for non-integer hop lengths (in samples).
    
    Args:
        x: the input audio waveform
        N: the FFT size
        hop_length: the hop size specified in samples, can be a non-integer
        window: the window to apply to the analysis frames
    
    Returns:
        X: the computed STFT matrix
        frame_offsets: a list specifying the offsets (in samples) of each analysis frame
    '''
    
    assert len(window) == N
    
    # get analysis frames
    numFrames = int((len(x) - N) // hop_length) + 1
    analysisFrames = np.zeros((numFrames, N))
    frame_offsets = np.rint(np.arange(numFrames) * hop_length).astype(int)
    for i, offset in enumerate(frame_offsets):
        analysisFrames[i,:] = x[offset: offset + N]

    # compute STFT
    analysisFrames = analysisFrames * window.reshape((1,-1))
    Xfull = np.fft.fft(analysisFrames, axis=1)
    halfLen = N//2 + 1
    X = Xfull[:,0:halfLen].T

    return X, frame_offsets

def estimateIF(S, sr, hop_samples):
    '''Estimates the instantaneous frequencies in a STFT matrix.
    
    This function is not actually used in our custom implementation of the phase vocoder -- 
    it is included here as a contrast to estimateIF_var() below.
    
    Args:
        S: the STFT matrix, should only contain the lower half of the frequency bins
        sr: sampling rate
        hop_samples: the hop size of the STFT analysis in samples
    
    Returns:
      A matrix containing the estimated instantaneous frequency at each time-frequency bin. This matrix should contain one less column than S.
    '''
    hop_sec = hop_samples / sr
    fft_size = (S.shape[0] - 1) * 2
    w_nom = np.arange(S.shape[0]) * sr / fft_size * 2 * np.pi
    w_nom = w_nom.reshape((-1,1))    
    unwrapped = np.angle(S[:,1:]) - np.angle(S[:,0:-1]) - w_nom * hop_sec
    wrapped = (unwrapped + np.pi) % (2 * np.pi) - np.pi
    w_if = w_nom + wrapped / hop_sec
    return w_if

def tsm_overlap_add(x, alpha = 1.0, L = 220):
    '''Time stretches the input signal using the overlap-add method.
    
    Uses a synthesis hop size that is half the value of L. 
    Note that this implementation allows for a non-integer analysis hop size (in samples), which
    ensures that the time-scale modification factor is exactly as specified.
    
    Args:
        x: the input signal
        alpha: the time stretch factor, which is defined as the ratio of the synthesis hop size to the analysis hop size
        L: the length of each analysis frame in samples
    
    Returns:
        the time-stretched signal y
    '''
    assert(L % 2 == 0), "Frame length must be even."
    Hs = L // 2
    
    # compute analysis frames
    Ha = Hs/alpha # allow non-integer analysis hop size
    numFrames = int((len(x) - L) // Ha) + 1
    analysisFrames = np.zeros((L, numFrames))
    for i in range(numFrames):
        offset = int(np.round(i * Ha))
        analysisFrames[:, i] = x[offset: offset + L]
    
    # reconstruction
    synthesisFrames = analysisFrames * hann_window(L).reshape((-1,1)) # use broadcasting
    y = np.zeros(Hs * (numFrames-1) + L)
    for i in range(numFrames):
        offset = i * Hs
        y[offset:offset+L] += synthesisFrames[:,i]
    
    return y

def mix_recordings(x1, x2):
    '''Mixes two audio waveforms together.
    
    Args:
        x1: first audio waveform
        x2: second audio waveform
    
    Returns:
        An audio waveform that is an average of the two waveforms. The length of the returned waveform is the minimum of the two lengths.
    '''
    min_length = min(len(x1), len(x2))
    y = .5 * (x1[0:min_length] + x2[0:min_length])
    return y

def tsm_hybrid(x, alpha=1.0, sr=22050):
    '''Time stretches the input signal using a hybrid method that combines overlap-add and phase vocoding.
    
    Args:
        x: the input signal
        alpha: the time stretch factor, which is defined as the ratio of the synthesis hop size to the analysis hop size
        sr: sampling rate
    
    Returns:
        the time-stretched signal y.
    '''
    
    xh, xp, _, _ = harmonic_percussive_separation(x)
    xh_stretched = tsm_phase_vocoder(xh, alpha)
    xp_stretched = tsm_overlap_add(xp, alpha)
    y = mix_recordings(xh_stretched, xp_stretched)
    
    return y

