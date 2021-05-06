
import torch
from torch import nn, optim
from torch.utils import data
from tqdm import trange, tqdm
from datetime import datetime
from tensorboardX import SummaryWriter

from module import DeepVoice3
from blocks import Entropy, GuidedAttentionLoss, Eve
from hyperparams import HyperParams as hp
from utils import griffinlim_collate_fn, \
                  preprocess_with_multi_process, \
                  init_worker_fn, \
                  NumpyDataset, \
                  load_model


def train_deepvoice3(load_latest: bool, path=None):
    writer = SummaryWriter("./log")

    # torch.autograd.set_detect_anomaly(True)
    dataset = NumpyDataset(metadata="./../dataset/LJSpeech-1.1/metadata.csv",
                           root="./../dataset/LJSpeech-1.1/preprocessed/",
                           vocab_to_id=hp.vocab_to_id,
                           id_to_vocab=hp.id_to_vocab)
    val_size = len(dataset) // 20
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = data.random_split(dataset, [train_size, val_size])
    train_dataloader = data.DataLoader(train_dataset,
                                       batch_size=hp.batch_size,
                                       shuffle=True,
                                       num_workers=5,
                                       pin_memory=True,
                                       collate_fn=griffinlim_collate_fn,
                                       worker_init_fn=init_worker_fn)
    val_dataloader = data.DataLoader(val_dataset,
                                     batch_size=hp.batch_size,
                                     shuffle=True,
                                     num_workers=2,
                                     pin_memory=True,
                                     collate_fn=griffinlim_collate_fn,
                                     worker_init_fn=init_worker_fn)

    device = "cuda"
    tts = DeepVoice3()
    global_step = 0
    optimizer = optim.Adam(
            tts.parameters(),
            lr=hp.lr,
            eps=hp.eps,
            weight_decay=hp.weight_decay,
            amsgrad=hp.use_ams_grad)

    if load_latest:
        tts = load_model(
                tts,
                # global_step=global_step,
                # optimizer=optimizer,
                path=path)
    tts = tts.to(device)

    mel_criterion = nn.L1Loss()
    mel_bce_criterion = nn.BCELoss()
    mag_criterion = nn.L1Loss()
    mag_bce_criterion = nn.BCELoss()
    done_criterion = nn.BCELoss()
    attn_criterion = GuidedAttentionLoss()
    # entropy_criterion = Entropy()

    # nn.utils.clip_grad_norm_(tts.parameters(), hp.max_gradient_norm)
    # nn.utils.clip_grad_value_(tts.parameters(), hp.max_gradient_value)

    for epoch in trange(hp.num_epoch, desc="epoch"):
        train_global_mel_loss = 0.0
        train_global_mag_loss = 0.0
        train_global_done_loss = 0.0
        train_global_attn_loss = 0.0
        train_global_loss = 0.0

        def train():
            nonlocal train_dataloader
            nonlocal optimizer
            tts.train()
            for label_mel_db, label_mag_db, label_done, frame_mask, \
                    script, script_mask in tqdm(train_dataloader):

                optimizer.zero_grad()

                zeros = torch.zeros(
                        (label_mel_db.size(0), hp.reduction_factor, hp.mel_bands)
                        )
                label_mel_db = torch.cat([zeros, label_mel_db], dim=1).to(device)
                label_mag_db = label_mag_db.to(device)
                frame_mask = frame_mask.unsqueeze(2).to(device)
                script_mask = script_mask.unsqueeze(2).to(device)
                mel, mag, done, attn = tts(
                        input=script.to(device),
                        decoder_input=label_mel_db[:, :-hp.reduction_factor],
                        frame_mask=frame_mask,
                        script_mask=script_mask)
                mel.masked_fill_(frame_mask, 0.0)
                mel_l1_loss = (1 - hp.mel_bce_rate) * \
                        mel_criterion(mel, label_mel_db[:, hp.reduction_factor:])
                if hp.mel_bce_rate == 0.0:
                    mel_bce_loss = None
                else:
                    mel_bce_loss = hp.mel_bce_rate *  \
                            mel_bce_criterion(mel, label_mel_db[:, hp.reduction_factor:])

                mag.masked_fill_(frame_mask, 0.0)
                mag_l1_loss = (1 - hp.mag_bce_rate) * \
                        mag_criterion(mag, label_mag_db)
                if hp.mag_bce_rate == 0.0:
                    mag_bce_loss = None
                else:
                    mag_bce_loss = hp.mag_bce_rate *  \
                            mag_bce_criterion(mag, label_mag_db)

                done_loss = done_criterion(done.squeeze(2), label_done.to(device))

                attn = torch.mean(torch.stack(attn), dim=0) \
                            .masked_fill_(frame_mask, 0.0)

                loss = mel_l1_loss + mag_l1_loss + done_loss
                if mel_bce_loss is not None:
                    loss += mel_bce_loss
                if mag_bce_loss is not None:
                    loss += mag_bce_loss
                if hp.use_guided > epoch:
                    attn_loss = attn_criterion(attn)  # + hp.entropy_rate * entropy_criterion(attn)
                    loss += attn_loss

                loss.backward()
                optimizer.step(lambda: loss)
                nonlocal global_step

                nonlocal train_global_mel_loss, train_global_mag_loss, \
                    train_global_done_loss, train_global_attn_loss, \
                    train_global_loss
                train_global_mel_loss += mel_l1_loss.item()
                if mel_bce_loss is not None:
                    train_global_mel_loss += mel_bce_loss.item()
                train_global_mag_loss += mag_l1_loss.item()
                if mag_bce_loss is not None:
                    train_global_mag_loss += mag_bce_loss.item()
                train_global_done_loss += done_loss.item()
                if hp.use_guided > epoch:
                    train_global_attn_loss += attn_loss.item()
                train_global_loss += loss.item()

                global_step += 1

                if global_step % hp.train_update_data_per == 0:
                    attn = attn[0]
                    writer.add_scalar(
                            "train mel l1_loss",
                            mel_l1_loss.item(),
                            global_step)
                    if mel_bce_loss is not None:
                        writer.add_scalar(
                                "train mel bce_loss",
                                mel_bce_loss.item(),
                                global_step)
                    writer.add_scalar(
                            "train mag l1_loss",
                            mag_l1_loss.item(),
                            global_step)
                    if mag_bce_loss is not None:
                        writer.add_scalar(
                                "train mag bce_loss",
                                mag_bce_loss.item(),
                                global_step)
                    writer.add_scalar(
                            "train done bce_loss",
                            done_loss.item(),
                            global_step)
                    if hp.use_guided > epoch:
                        writer.add_scalar(
                                "train attn loss",
                                attn_loss.item(),
                                global_step)

                    attn.transpose_(0, 1)
                    writer.add_image(
                            "train attention",
                            torch.cat([attn, attn.sum(dim=0).unsqueeze(0)], dim=0),
                            global_step=epoch,
                            dataformats="HW")
                    writer.add_image(
                            "train mel spectrogram",
                            mel[0].transpose(0, 1),
                            global_step=epoch,
                            dataformats="HW")
                    writer.add_image(
                            "train label mel spectrogram",
                            label_mel_db[0].transpose(0, 1),
                            global_step=epoch,
                            dataformats="HW")

                    writer.add_scalar("loss", loss.item(), global_step)
        train()

        if epoch % 5 == 0:
            val_global_mel_loss = 0.0
            val_global_mag_loss = 0.0
            val_global_done_loss = 0.0
            val_global_attn_loss = 0.0
            val_global_loss = 0.0

            def val():
                nonlocal val_dataloader
                val_step = 0
                with torch.no_grad():
                    tts.eval()
                    for label_mel_db, label_mag_db, label_done, frame_mask, \
                            script, script_mask in tqdm(val_dataloader):

                        val_step += 1
                        zeros = torch.zeros(
                                (label_mel_db.size(0), hp.reduction_factor, hp.mel_bands)
                                )
                        label_mel_db = torch.cat([zeros, label_mel_db], dim=1).to(device)
                        label_mag_db = label_mag_db.to(device)
                        frame_mask = frame_mask.unsqueeze(2).to(device)
                        script_mask = script_mask.unsqueeze(2).to(device)
                        mel, mag, done, attn = tts(
                                input=script.to(device),
                                frame_mask=frame_mask,
                                script_mask=script_mask,
                                decoder_input=label_mel_db[:, :-hp.reduction_factor])
                        mel.masked_fill_(frame_mask, hp.eps)
                        mag.masked_fill_(frame_mask, hp.eps)

                        mel_l1_loss = (1 - hp.mel_bce_rate) * \
                                mel_criterion(mel, label_mel_db[:, hp.reduction_factor:])
                        if hp.mel_bce_rate == 0.0:
                            mel_bce_loss = None
                        else:
                            mel_bce_loss = hp.mel_bce_rate *  \
                                    mel_bce_criterion(mel, label_mel_db[:, hp.reduction_factor:])

                        mag_l1_loss = (1 - hp.mag_bce_rate) * \
                                mag_criterion(mag, label_mag_db)
                        if hp.mag_bce_rate == 0.0:
                            mag_bce_loss = None
                        else:
                            mag_bce_loss = hp.mag_bce_rate *  \
                                    mag_bce_criterion(mag, label_mag_db)

                        done_loss = done_criterion(done.squeeze(2), label_done.to(device))

                        attn = torch.mean(torch.stack(attn), dim=0) \
                                    .masked_fill_(frame_mask, hp.eps)

                        loss = mel_l1_loss + mag_l1_loss + done_loss
                        if mel_bce_loss is not None:
                            loss += mel_bce_loss
                        if mag_bce_loss is not None:
                            loss += mag_bce_loss
                        if hp.use_guided > epoch:
                            attn_loss = attn_criterion(attn)  # + hp.entropy_rate * entropy_criterion(attn)
                            loss += attn_loss

                        nonlocal val_global_mel_loss, val_global_mag_loss, \
                            val_global_done_loss, val_global_attn_loss, \
                            val_global_loss
                        val_global_mel_loss += mel_l1_loss.item()
                        if mel_bce_loss is not None:
                            val_global_mel_loss += mel_bce_loss.item()
                        val_global_mag_loss += mag_l1_loss.item()
                        if mag_bce_loss is not None:
                            val_global_mag_loss += mag_bce_loss.item()
                        val_global_done_loss += done_loss.item()
                        if hp.use_guided > epoch:
                            val_global_attn_loss += attn_loss.item()
                        val_global_loss += loss.item()

                        if val_step % hp.eval_update_data_per == 0:
                            attn = attn[0]
                            attn.transpose_(0, 1)
                            writer.add_image(
                                    "val attention",
                                    torch.cat([attn, attn.sum(dim=0).unsqueeze(0)], dim=0),
                                    global_step=epoch,
                                    dataformats="HW")
                            writer.add_image(
                                    "val mel spectrogram",
                                    mel[0].transpose(0, 1),
                                    global_step=epoch,
                                    dataformats="HW")
                            writer.add_image(
                                    "val label mel spectrogram",
                                    label_mel_db[0, hp.reduction_factor:].transpose(0, 1),
                                    global_step=epoch,
                                    dataformats="HW")
                            writer.add_image(
                                    "val mag spectrogram",
                                    mag[0].transpose(0, 1),
                                    global_step=epoch,
                                    dataformats="HW")
                            writer.add_image(
                                    "val label mag spectrogram",
                                    label_mag_db[0, hp.reduction_factor:].transpose(0, 1),
                                    global_step=epoch,
                                    dataformats="HW")
            val()
            writer.add_scalars("average loss", {
                "train": train_global_loss / len(train_dataloader),
                "validate": val_global_loss / len(val_dataloader), }, epoch)
            print(f"""
            global_step: {global_step}
                train
                -----------------------------
                mel loss: {train_global_mel_loss / len(train_dataloader)}
                mag loss: {train_global_mag_loss / len(train_dataloader)}
                done loss: {train_global_done_loss / len(train_dataloader)}
                attn loss: {train_global_attn_loss / len(train_dataloader)}
                -----------------------------
                validation
                -----------------------------
                mel loss: {val_global_mel_loss / len(val_dataloader)}
                mag loss: {val_global_mag_loss / len(val_dataloader)}
                done loss: {val_global_done_loss / len(val_dataloader)}
                attn loss: {val_global_attn_loss / len(val_dataloader)}
                -----------------------------

                -----------------------------
                train loss: {train_global_loss / len(train_dataloader)}
                val loss: {val_global_loss / len(val_dataloader)}
                """)
            t = datetime.today()
            torch.save(tts.state_dict(), f"./model/{t}.model")
            torch.save({
                    "global_step": global_step,
                    "optimizer": optimizer.state_dict(),
                    "param": hp
                }, f"./model/{t}.param")


if __name__ == "__main__":
    # preprocess
    preprocess_with_multi_process(force=False)
    # training
    train_deepvoice3(load_latest=False)

