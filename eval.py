import sys

import torch
import numpy as np
from tqdm import trange

from module import DeepVoice3
from utils import griffinlim_collate_fn, \
                  preprocess_with_multi_process, \
                  init_worker_fn, \
                  NumpyDataset, \
                  load_model
from hyperparams import HyperParams as hp
from waveutils import magnitude_db_save_as_wav

from torch.utils import data


device = "cpu"


def evaluate(sentence: str, path=None):
    tts = DeepVoice3()
    tts = load_model(tts, path=path).to(device).eval()

    input = torch.tensor([[hp.vocab_to_id[c]
                          for c in sentence
                          if c in hp.id_to_vocab]]).to(device)

    decoder_init_state = torch.zeros(
        (1, hp.reduction_factor, hp.mel_bands)
        ).to(device)

    with torch.no_grad():
        _, mag, _, _ = tts(input=input,
                           init_state=decoder_init_state)
        mag = mag.squeeze(0).detach().cpu().numpy()
        magnitude_db_save_as_wav(mag, "/tmp/hoge.wav")


def test():
    dataset = NumpyDataset(metadata="./../dataset/LJSpeech-1.1/metadata.csv",
                           root="./../dataset/LJSpeech-1.1/preprocessed/",
                           vocab_to_id=hp.vocab_to_id,
                           id_to_vocab=hp.id_to_vocab)
    train_dataloader = data.DataLoader(dataset,
                                       batch_size=1,
                                       shuffle=True,
                                       num_workers=1,
                                       # pin_memory=True,
                                       collate_fn=griffinlim_collate_fn,
                                       worker_init_fn=init_worker_fn)
    tts = DeepVoice3()
    tts = load_model(tts).to(device).eval()
    for label_mel_db, label_mag_db, label_done, frame_mask, \
            script, script_mask in train_dataloader:
        zeros = torch.zeros(
                (label_mel_db.size(0), hp.reduction_factor, hp.mel_bands)
                )
        label_mel_db = torch.cat([zeros, label_mel_db], dim=1).to(device)
        frame_mask = frame_mask.unsqueeze(2).to(device)
        script_mask = script_mask.unsqueeze(2).to(device)
        mel, mag, done, attn = tts(
                input=script.to(device),
                decoder_input=label_mel_db[:, :-hp.reduction_factor],
                frame_mask=frame_mask,
                script_mask=script_mask)
        # from matplotlib import pyplot as plt
        # mel_ = mel.squeeze(0).detach().numpy()
        # plt.imshow(mel_.T)
        # plt.show()
        mag = mag.squeeze(0).detach().cpu().numpy()
        magnitude_db_save_as_wav(mag, "/tmp/hoge.wav")
        break

if __name__ == "__main__":
    hp.mode = "generate"
    if len(sys.argv) > 1:
        path = sys.argv[1]
        while True:
            evaluate(input(), path)
    else:
        path=None
        evaluate("there's a way to measure the acute emotional intelligence that has never gone out of style.", path)
        # evaluate("icassp stands for the international conference on acoustics, speech and signal processing.", path)
        # evaluate("printing, in the only sense with which we are at present concerned, differs from most if not from all the arts and crafts represented in the Exhibition.")
    # test()
