
from torchaudio import functional as AF
import librosa
import soundfile
import numpy as np

from hyperparams import HyperParams as hp


def get_spectgrams(path: str):
    '''
    Returns:
        mel_db [timestep, mel_bands]
        magnitude_db [timestep, 1 + fft_size // 2]
        done [timestep]:
            end flag
            last element of `done` is only set 1, otherwise 0
    '''
    y, sr = librosa.load(path=path, sr=hp.sample_rate)
    y, _ = librosa.effects.trim(y, top_db=hp.top_db, ref=hp.ref_db)
    y = librosa.effects.preemphasis(y, coef=hp.preemphasis_coef)
    data = librosa.stft(
            y,
            n_fft=hp.fft_size,
            win_length=hp.fft_window_size,
            hop_length=hp.fft_window_shift,
            window=hp.fft_window_kind)
    linear = np.abs(data).astype(np.float32)

    mel = np.dot(librosa.filters.mel(sr=hp.sample_rate,
                                     n_fft=hp.fft_size,
                                     n_mels=hp.mel_bands,
                                     dtype=np.float32), linear)
    mel_db = librosa.amplitude_to_db(mel,
                                     top_db=hp.top_db,
                                     ref=hp.ref_db).T
    mel_db = np.clip(
            (mel_db - hp.ref_db + hp.top_db) / hp.top_db, 0.0, 1.0)

    done = np.zeros_like(mel[0, :])
    done[-1] = 1

    magnitude_db = librosa.amplitude_to_db(linear,
                                           top_db=hp.top_db,
                                           ref=hp.ref_db).T

    # normalize
    magnitude_db = np.clip(
            (magnitude_db - hp.ref_db + hp.top_db) / hp.top_db, 0.0, 1.0)

    return mel_db, magnitude_db, done


def magnitude_db_to_wav(magnitude_db):
    magnitude = librosa.db_to_amplitude(magnitude_db.T, ref=hp.ref_db)
    return librosa.griffinlim(magnitude,
                              win_length=hp.fft_window_size,
                              hop_length=hp.fft_window_shift,
                              window=hp.fft_window_kind)


def magnitude_db_save_as_wav(magnitude_db, path: str):
    # de-normalize
    magnitude_db = magnitude_db * hp.top_db - hp.top_db + hp.ref_db
    wav = magnitude_db_to_wav(magnitude_db)
    soundfile.write(path, wav, samplerate=hp.sample_rate)

