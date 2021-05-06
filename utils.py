
import torch
from torch import nn
from torch.utils import data
import torchvision
import torchaudio
from torchaudio import functional as AF
from glob import glob
from tqdm import tqdm
import os
import shutil
from multiprocessing import Pool
import numpy as np
import csv
from typing import List, Dict, Any
import datetime
import librosa

from waveutils import get_spectgrams

from hyperparams import HyperParams as hp


if hp.corpus == "ljspeech":
    preprocessed_dir = "./../dataset/LJSpeech-1.1/preprocessed/"
    wavs_dir = "./../dataset/LJSpeech-1.1/wavs/"
elif hp.corpus == "jsut":
    preprocessed_dir = "./../dataset/jsut_ver1.1/basic5000/preprocessed/"
    wavs_dir = "./../dataset/jsut_ver1.1/basic5000/wav/"
elif hp.corpus == "vctk":
    pass


def _process(path):
    mel_db, magnitude_db, done = get_spectgrams(path)
    path, ext = os.path.splitext(path)
    fname = os.path.basename(path)
    save_path = os.path.join(preprocessed_dir, fname + ".npy")
    np.save(save_path, [mel_db.T, magnitude_db.T, done.T])


def preprocess(force=False):
    if os.path.isdir(preprocessed_dir):
        if force:
            shutil.rmtree(preprocessed_dir)
            os.mkdir(preprocessed_dir)
        else:
            return
    else:
        os.mkdir(preprocessed_dir)

    for path in tqdm(glob(os.path.join(wavs_dir, "*"))):
        _process(path)


def preprocess_with_multi_process(force=False):
    if os.path.isdir(preprocessed_dir):
        if force:
            shutil.rmtree(preprocessed_dir)
            os.mkdir(preprocessed_dir)
        else:
            return
    else:
        os.mkdir(preprocessed_dir)
    pool = Pool()
    files = [path for path in glob(wavs_dir + "*")]
    with tqdm(total=len(files)) as t:
        for _ in pool.imap_unordered(_process, files):
            t.update(1)


class NumpyDataset(data.Dataset):
    def __init__(self,
                 metadata,
                 root,
                 vocab_to_id: Dict[str, int],
                 id_to_vocab: List[str],
                 kind="griffinlim",
                 corpus="ljspeech",
                 transform=None):
        self.root = root
        self.files = []
        self.vocab_to_id = vocab_to_id
        self.id_to_vocab = id_to_vocab
        self.kind = kind
        self.transform = transform
        if corpus == "ljspeech":
            with open(metadata, newline="") as f:
                reader = csv.reader(f, delimiter="|", quotechar="$")
                for row in reader:
                    fname, script, normalized = row
                    script = script.lower()
                    normalized = normalized.lower()
                    self.files.append((fname, normalized))
        elif corpus == "jsut":
            with open(metadata, newline="") as f:
                for row in f.readline():
                    fname, script = row.split(":")[0:2]
                    # TODO must convert script to hiragana expression
                    self.files.append((fname, script))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        fname, script = self.files[idx]
        script = np.array([self.vocab_to_id[c]
                           for c in script if c in self.id_to_vocab])
        path = os.path.join(self.root, fname + ".npy")
        if self.kind == "griffinlim":
            mel, mag, done = np.load(path, allow_pickle=True)
            length = mel.shape[1]
            ideal_length = np.ceil(
                length / hp.upsample_rate).astype(np.int) * hp.upsample_rate
            shortage = ideal_length - length
            mel = np.pad(mel, [(0, 0), (0, shortage)])
            mel = torch.from_numpy(mel)
            mag = np.pad(mag, [(0, 0), (0, shortage)])
            mag = torch.from_numpy(mag)
            done = np.pad(done, [(0, shortage)],
                          constant_values=hp.done_out_of_range)
            done = torch.from_numpy(done)
            frame_mask = done.new_full(done.size(), False, dtype=torch.bool)

            script = torch.from_numpy(script)
            script_mask = script.new_full(script.size(), False, dtype=torch.bool)

            return mel, mag, done, frame_mask, script, script_mask
        else:
            assert False


class DeepVoice3Dataset(data.Dataset):
    def __init__(self,
                 metadata,
                 root,
                 vocab_to_id: Dict[str, int],
                 id_to_vocab: List[str],
                 transform="griffinlim",
                 device="cpu"):
        self.root = root
        self.files = []
        self.vocab_to_id = vocab_to_id
        self.id_to_vocab = id_to_vocab
        self.transform = transform
        self.to_mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=hp.sample_rate,
            n_fft=hp.fft_size,
            win_length=hp.fft_window_size,
            hop_length=hp.fft_window_shift,
            n_mels=hp.mel_bands).to(device, non_blocking=True)
        if transform == "wavenet":
            self.mu_law_encode = torchaudio.transforms.MuLawDecoding(
                quantization_channels=hp.wavenet_quantization_channel).to(device, non_blocking=True)
        elif transform == "griffinlim":
            self.spectrogram = torchaudio.transforms.Spectrogram(
                    n_fft=hp.fft_size,
                    win_length=hp.fft_window_size,
                    hop_length=hp.fft_window_shift).to(device, non_blocking=True)
            self.to_mag = torchaudio.transforms.AmplitudeToDB(
                    stype="magnitude",
                    top_db=hp.top_db).to(device, non_blocking=True)
        self.device = device
        with open(metadata, newline="") as f:
            reader = csv.reader(f, delimiter="|", quotechar="$")
            for row in reader:
                fname, script, normalized = row
                script = script.lower()
                normalized = normalized.lower()
                self.files.append((fname, normalized))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        fname, script = self.files[idx]
        script = np.array([self.vocab_to_id[c]
                           for c in script if c in self.id_to_vocab])
        script = torch.from_numpy(script).to(self.device, non_blocking=True)
        path = os.path.join(self.root, fname + ".wav")
        # data, rate = torchaudio.load(path)
        data, _ = librosa.load(path, sr=hp.sample_rate)
        data = torch.from_numpy(data).to(self.device, non_blocking=True)

        if self.transform == "wavenet":
            mel = self.to_mel(data)
            mu_law_encoded = self.mu_law_encode(mel)
            done = torch.zeros(mel.shape).to(self.device, non_blocking=True)
            return mel, mu_law_encoded, done, script
        elif self.transform == "griffinlim":
            mel = self.to_mel(data)

            data = self.spectrogram(data)
            mag = self.to_mag(data)

            seq_length = mel.size(1)
            done = torch.zeros((1, seq_length)).to(self.device, non_blocking=True)
            done[-1] = 1.0
            return mel, mag, done, script


def griffinlim_collate_fn(batch: List[Any]):
    buf_mel_db = []
    buf_magnitude_db = []
    buf_done = []
    buf_frame_mask = []
    buf_script = []
    buf_script_mask = []
    for mel_db, magnitude_db, done, frame_mask, script, script_mask in batch:
        buf_mel_db.append(mel_db.t())
        buf_magnitude_db.append(magnitude_db.t())
        buf_done.append(done.t())
        buf_frame_mask.append(frame_mask.t())
        buf_script.append(script.t())
        buf_script_mask.append(script_mask.t())
    return \
        torch.nn.utils.rnn.pad_sequence(
                buf_mel_db, batch_first=True), \
        torch.nn.utils.rnn.pad_sequence(
                buf_magnitude_db, batch_first=True), \
        torch.nn.utils.rnn.pad_sequence(
                buf_done, batch_first=True, padding_value=hp.done_out_of_range), \
        torch.nn.utils.rnn.pad_sequence(
                buf_frame_mask, batch_first=True, padding_value=True), \
        torch.nn.utils.rnn.pad_sequence(
                buf_script, batch_first=True), \
        torch.nn.utils.rnn.pad_sequence(
                buf_script_mask, batch_first=True, padding_value=True)


def init_worker_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def init_weight(weight):
    if hp.init_distribution == "normal":
        nn.init.normal_(weight, 0.1)
    elif hp.init_distribution == "uniform":
        nn.init.uniform_(weight, 0.0, 0.1)
    elif hp.init_distribution == "one":
        nn.init.constant_(weight, 1.0)
    elif hp.init_distribution == "zero":
        nn.init.constant_(weight, 0.0)
    elif hp.init_distribution == "xavier":
        nn.init.xavier_normal_(weight)
    elif hp.init_distribution == "he":
        nn.init.kaiming_normal_(weight)


def load_model(tts, global_step=None, optimizer=None, path=None):
    if path is not None:
        state = torch.load(path, map_location=lambda storage, loc: storage)
        tts.load_state_dict(state)
        return tts
    path = glob("model/*")
    path = list(map(lambda p: os.path.splitext(p)[0], path))
    path = list(map(lambda p: os.path.basename(p), path))
    date = list(map(lambda d: datetime.datetime.fromisoformat(d), path))
    latest = sorted(date)[-1]
    print(latest)
    model = torch.load(f"model/{latest}.model", map_location=lambda storage, loc: storage)
    if optimizer is not None or global_step is not None:
        param = torch.load(f"model/{latest}.param")
    tts.load_state_dict(model)
    if global_step is not None:
        global_step = param["global_step"]
    if optimizer is not None:
        optimizer.load_state_dict(param["optimizer"])
    if global_step is None and optimizer is None:
        return tts
    else:
        return tts, global_step, optimizer


def depend_factor(*args):
    size = len(args)
    result = 1
    for i in range(size):
        result += (args[size - i - 1] - 1) * (size - i)
    return result
