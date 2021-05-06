
import torch
from torch import nn, cuda
from torch.nn import functional as F
from tqdm import trange

from hyperparams import HyperParams as hp
from blocks import DecoderBlock, \
                   FCBlock, \
                   ConvTransBlock, \
                   ConvBlockE, \
                   PoolBlock, \
                   Transpose, \
                   TransformerDecoderLayer, \
                   PositionalEncoding

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(
                num_embeddings=hp.vocab_size,
                embedding_dim=hp.charactor_embedding_size,
                padding_idx=hp.vocab_to_id["\0"])
        if hp.speaker_size != 1:
            self.fc_speaker1 = FCBlock(
                    in_features=hp.speaker_embedding,
                    out_features=hp.charactor_embedding_size,
                    use_norm=hp.encoder_norm)
            self.fc_speaker2 = FCBlock(
                    in_features=hp.speaker_embedding,
                    out_features=hp.charactor_embedding_size,
                    use_norm=hp.encoder_norm)
        self.conv_blocks = nn.Sequential(
            Transpose(1, 2),
            ConvBlockE(
                in_channels=hp.charactor_embedding_size,
                out_channels=hp.encoder_conv_channels,
                kernel_size=1,
                use_norm=hp.encoder_norm,
                glu=False,
                causal=False),
            *[ConvBlockE(
                in_channels=hp.encoder_conv_channels,
                out_channels=hp.encoder_conv_channels,
                kernel_size=hp.encoder_conv_width,
                dilation=hp.encoder_dilation_base ** (i % 4),
                activation="none",
                use_norm=hp.encoder_norm,
                causal=False)
                for i in range(hp.encoder_layer_size)],
            ConvBlockE(
                in_channels=hp.encoder_conv_channels,
                out_channels=hp.charactor_embedding_size,
                kernel_size=1,
                glu=False,
                use_norm=hp.encoder_norm,
                causal=False),
            Transpose(1, 2))

    def forward(self, input, speaker_embedding=None):
        if hp.debug:
            print("Encoder",
                    torch.isnan(input).any().item(),
                    torch.isinf(input).any().item())
        embedded = self.embedding(input)

        if hp.speaker_size != 1:
            if speaker_embedding is None:
                assert False, "multi speaker requires speaker_embedding"
            speaker_out1 = self.fc_speaker1(speaker_embedding)
            speaker_out1 = F.softsign(speaker_out1)
            embedded += speaker_out1

        # [batch_size, sentence_length, charactor_embedding_size]
        x = self.conv_blocks(embedded)

        if hp.speaker_size != 1:
            if speaker_embedding is None:
                assert False, "multi speaker requires speaker_embedding"
            speaker_out2 = self.fc_speaker2(speaker_embedding)
            speaker_out2 = F.softsign(speaker_out1)
            x += speaker_out2

        keys = x
        values = (x + embedded) * 0.5 ** 0.5
        return keys, values


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # prenet
        if hp.speaker_size != 1:
            # TODO
            # self.fc_speaker = FCBlock(hp.speaker_embedding, )
            pass
        self.prenet = nn.Sequential(
            Transpose(1, 2),

            ConvBlockE(
                in_channels=hp.mel_bands,
                out_channels=hp.decoder_prenet_affine_size,
                kernel_size=1,
                autoregressive=True,
                use_norm=hp.decoder_norm,
                dp=hp.decoder_dp,
                glu=False,
                causal=True),
            *[ConvBlockE(
                in_channels=hp.decoder_prenet_affine_size,
                out_channels=hp.decoder_prenet_affine_size,
                kernel_size=hp.decoder_prenet_conv_width,
                activation="none",
                autoregressive=True,
                use_norm=hp.decoder_norm,
                dp=hp.decoder_dp,
                dilation=hp.decoder_pre_dilation_base ** (i % 4),
                causal=True)
                for i in range(hp.decoder_prenet_layer_size)],
            Transpose(1, 2))
        # decoder block
        self.decoder_list = nn.ModuleList([
            DecoderBlock(
                kernel_size=hp.decoder_conv_width,
                channels=hp.decoder_prenet_affine_size,
                keys_values_size=hp.charactor_embedding_size,
                conv_layers=hp.decoder_conv_layer_size[i],
                dilation=hp.decoder_dilation_base,
                hidden_size=hp.attention_hidden_size,
                autoregressive=True,
                use_norm=hp.decoder_norm,
                num_head=hp.attention_num_head)
            for i in range(hp.decoder_layer_size)])

        self.postnet = nn.Sequential(
            Transpose(1, 2),
            ConvBlockE(
                in_channels=hp.decoder_prenet_affine_size,
                out_channels=hp.decoder_postnet_affine_size,
                kernel_size=1,
                glu=False,
                autoregressive=True,
                use_norm=hp.decoder_norm,
                dp=hp.decoder_dp,
                causal=True),
            *[ConvBlockE(
                    in_channels=hp.decoder_postnet_affine_size,
                    out_channels=hp.decoder_postnet_affine_size,
                    kernel_size=hp.decoder_postnet_conv_width,
                    activation="none",
                    autoregressive=True,
                    causal=True,
                    use_norm=hp.decoder_norm,
                    dp=hp.decoder_dp,
                    dilation=hp.decoder_post_dilation_base ** (i % 4))
                for i in range(hp.decoder_postnet_layer_size)],
            ConvBlockE(
                in_channels=hp.decoder_postnet_affine_size,
                out_channels=hp.decoder_postnet_affine_size,
                kernel_size=hp.decoder_postnet_conv_width,
                glu=False,
                autoregressive=True,
                use_norm=hp.decoder_norm,
                dp=hp.decoder_dp,
                causal=True))

        self.fc_done = nn.Sequential(
            FCBlock(
                in_features=hp.decoder_postnet_affine_size,
                out_features=1,
                use_norm=hp.decoder_norm,
                activation="sigmoid"))

        self.conv_mel = nn.Sequential(
            ConvBlockE(
                in_channels=hp.decoder_postnet_affine_size,
                out_channels=hp.mel_bands,
                kernel_size=1,
                glu=False,
                use_norm=hp.decoder_norm if hp.decoder_norm != "batch" else "weight",
                causal=True,
                activation="sigmoid"),
            Transpose(1, 2))

    def forward(self,
                # [batch_size, frame_langth, mel_bands] input,
                input,
                # [batch_size, sentence_length, charactor_embedding_size]
                keys, values,
                timestep,
                target_mask=None,
                src_mask=None,
                speaker_embedding=None):
        '''
        done [batch_size, reduction_factor, 1]
        mel [batch_size, reduction_factor, decoder_prenet_affine_size]
        x [batch_size, reduction_factor, decoder_prenet_affine_size]
        '''
        if hp.debug:
            print("Decoder",
                  torch.isnan(input).any().item(),
                  torch.isinf(input).any().item(),
                  torch.isnan(keys).any().item(),
                  torch.isinf(keys).any().item(),
                  torch.isnan(values).any().item(),
                  torch.isinf(values).any().item())

        # PreNet
        x = self.prenet(input)

        # DecoderBlock
        attention_list = []
        for layer in self.decoder_list:
            x, attention = layer(input=x,
                                 keys=keys,
                                 values=values,
                                 timestep=timestep,
                                 target_mask=target_mask,
                                 src_mask=src_mask)
            attention_list.append(attention)

        x = self.postnet(x)

        # Mel Output
        mel = self.conv_mel(x)

        x.transpose_(1, 2)

        # Done
        done = self.fc_done(x)

        return done, mel, x, attention_list


class Converter(nn.Module):
    def __init__(self):
        super(Converter, self).__init__()
        self.conv_list = nn.Sequential(
            Transpose(1, 2),
            ConvBlockE(
                in_channels=hp.converter_input_channel_size,
                out_channels=hp.converter_pre_conv_channels,
                kernel_size=1,
                causal=False,
                use_norm=hp.converter_norm,
                length_shrink_rate=4,
                pool_kind=hp.converter_pool_kind,
                pool_kernel_size=hp.converter_pool_kernel_size,
                dp=hp.converter_dp,
                glu=False),
            *[ConvBlockE(
                in_channels=hp.converter_pre_conv_channels,
                out_channels=hp.converter_pre_conv_channels,
                kernel_size=hp.converter_conv_width,
                dilation=hp.converter_dilation_base ** (i % 2),
                activation="none",
                use_norm=hp.converter_norm,
                dp=hp.converter_dp,
                causal=False)
                for i in range(hp.converter_pre_layer_size)],
            ConvTransBlock(
                in_channels=hp.converter_pre_conv_channels,
                out_channels=hp.converter_conv_channels,
                kernel_size=hp.converter_conv_trans_width,
                length_expand_rate=2,
                dp=hp.converter_dp,
                use_norm=hp.converter_norm),
            *[ConvBlockE(
                in_channels=hp.converter_conv_channels,
                out_channels=hp.converter_conv_channels,
                kernel_size=hp.converter_conv_width,
                dilation=hp.converter_dilation_base ** (i % 2),
                activation="none",
                use_norm=hp.converter_norm,
                dp=hp.converter_dp,
                causal=False)
                for i in range(hp.converter_layer_size)],
            ConvTransBlock(
                in_channels=hp.converter_conv_channels,
                out_channels=hp.converter_post_conv_channels,
                kernel_size=hp.converter_conv_trans_width,
                length_expand_rate=2,
                dp=hp.converter_dp,
                use_norm=hp.converter_norm),
            *[ConvBlockE(
                in_channels=hp.converter_post_conv_channels,
                out_channels=hp.converter_post_conv_channels,
                kernel_size=hp.converter_conv_width,
                dilation=hp.converter_dilation_base ** (i % 2),
                activation="none",
                use_norm=hp.converter_norm,
                dp=hp.converter_dp,
                causal=False)
                for i in range(hp.converter_post_layer_size)],
            ConvBlockE(
                in_channels=hp.converter_post_conv_channels,
                out_channels=hp.fft_size // 2 + 1,
                kernel_size=hp.converter_conv_width,
                causal=False,
                dp=hp.converter_dp,
                activation="none",
                use_norm=hp.converter_norm),
            ConvBlockE(
                in_channels=hp.fft_size // 2 + 1,
                out_channels=hp.fft_size // 2 + 1,
                kernel_size=hp.converter_conv_width,
                causal=False,
                use_norm=hp.converter_norm,
                dp=hp.converter_dp,
                glu=False,
                activation="sigmoid"),
            )

    def forward(self, input):
        if hp.debug:
            print("Converter",
                    torch.isnan(input).any().item(),
                    torch.isinf(input).any().item(),
                    )
        mag = self.conv_list(input)
        return mag.transpose(1, 2)


class TransformerEncoder(nn.Module):
    def __init__(self):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Embedding(
                num_embeddings=hp.vocab_size,
                embedding_dim=hp.charactor_embedding_size,
                padding_idx=hp.vocab_to_id["\0"])
        self.prenet = nn.Sequential(
            Transpose(1, 2),
            ConvBlockE(
                in_channels=hp.charactor_embedding_size,
                out_channels=hp.transformer_encoder_d_model,
                kernel_size=1,
                use_norm=hp.encoder_norm,
                glu=True,
                activation="none",
                causal=False),
            *[ConvBlockE(
                in_channels=hp.transformer_encoder_d_model,
                out_channels=hp.transformer_encoder_d_model,
                kernel_size=hp.encoder_conv_width,
                use_norm=hp.encoder_norm,
                glu=True,
                activation="none",
                causal=False)
                for i in range(hp.transformer_encoder_prenet_layers)],
            Transpose(1, 2))
        self.pe = PositionalEncoding(
                max_length=hp.max_sentence_length,
                dim=hp.transformer_encoder_d_model,
                w=hp.w_key,
                auto=hp.position_encoding_auto,
                scaled=hp.position_encoding_scaled)
        encoder_layer = nn.TransformerEncoderLayer(
                d_model=hp.transformer_encoder_d_model,
                nhead=hp.transformer_num_head_encoder,
                dim_feedforward=hp.transformer_encoder_dim_feedforward,
                dropout=hp.dp)
        self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=hp.transformer_encoder_layers,
                norm=nn.LayerNorm(hp.transformer_encoder_d_model))

        self.postnet = nn.Sequential(
                Transpose(1, 2),
                *[ConvBlockE(
                    in_channels=hp.transformer_encoder_d_model,
                    out_channels=hp.transformer_encoder_d_model,
                    kernel_size=hp.encoder_conv_width,
                    glu=True,
                    dilation=hp.encoder_dilation_base ** (i % 4),
                    activation="none",
                    use_norm=hp.encoder_norm,
                    causal=False)
                    for i in range(hp.transformer_encoder_postnet_layers)],
                ConvBlockE(
                    in_channels=hp.transformer_encoder_d_model,
                    out_channels=hp.charactor_embedding_size,
                    kernel_size=1,
                    glu=True,
                    activation="none",
                    use_norm=hp.encoder_norm,
                    causal=False),
                Transpose(1, 2))

    def forward(self, input):
        embedded = self.embedding(input)
        x = self.prenet(embedded)
        x = self.pe(x)
        x = self.transformer_encoder(x)
        x = self.postnet(x)
        keys = x
        values = (x + embedded) * 0.5 ** 0.5
        return keys, values


class TransformerDecoder(nn.Module):
    def __init__(self):
        super(TransformerDecoder, self).__init__()
        self.prenet = nn.Sequential(
            Transpose(1, 2),

            ConvBlockE(
                in_channels=hp.mel_bands,
                out_channels=hp.decoder_prenet_affine_size,
                kernel_size=1,
                autoregressive=True,
                use_norm=hp.decoder_norm,
                dp=hp.decoder_dp,
                glu=True,
                activation="none",
                causal=True),
            *[ConvBlockE(
                in_channels=hp.decoder_prenet_affine_size,
                out_channels=hp.decoder_prenet_affine_size,
                kernel_size=hp.decoder_prenet_conv_width,
                autoregressive=True,
                use_norm=hp.decoder_norm,
                dp=hp.decoder_dp,
                glu=True,
                activation="none",
                dilation=hp.decoder_pre_dilation_base ** (i % 4),
                causal=True)
                for i in range(hp.decoder_prenet_layer_size)],
            ConvBlockE(
                in_channels=hp.decoder_prenet_affine_size,
                out_channels=hp.transformer_decoder_d_model,
                kernel_size=1,
                autoregressive=True,
                use_norm=hp.decoder_norm,
                dp=hp.decoder_dp,
                glu=True,
                activation="none",
                causal=True),
            Transpose(1, 2))
        self.pe = PositionalEncoding(
                max_length=hp.max_timestep,
                dim=hp.transformer_decoder_d_model,
                w=hp.w_query,
                auto=hp.position_encoding_auto,
                scaled=hp.position_encoding_scaled)
        self.transformer_decoder = nn.ModuleList([
            TransformerDecoderLayer(
                d_model=hp.transformer_decoder_d_model,
                nhead=hp.transformer_num_head_decoder,
                dim_feedforward=hp.transformer_decoder_dim_feedforward,
                keys_values_size=hp.charactor_embedding_size,
                use_norm=hp.decoder_norm,
                dp=hp.decoder_dp)
            for _ in range(hp.transformer_decoder_layers)])
        self.fc_done = nn.Sequential(
            FCBlock(
                in_features=hp.transformer_decoder_d_model,
                out_features=1,
                use_norm=hp.decoder_norm,
                activation="sigmoid"))

        self.postnet = nn.Sequential(
            *[ConvBlockE(
                in_channels=hp.transformer_decoder_d_model,
                out_channels=hp.transformer_decoder_d_model,
                kernel_size=hp.decoder_postnet_conv_width,
                autoregressive=True,
                use_norm=hp.decoder_norm,
                dp=hp.decoder_dp,
                glu=True,
                activation="none",
                dilation=hp.decoder_post_dilation_base ** (i % 4),
                causal=True)
                for i in range(hp.decoder_postnet_layer_size)])

        self.conv_mel = nn.Sequential(
            ConvBlockE(
                in_channels=hp.transformer_decoder_d_model,
                out_channels=hp.mel_bands,
                kernel_size=1,
                glu=False,
                autoregressive=True,
                use_norm=hp.decoder_norm,
                causal=True,
                dp=hp.decoder_dp,
                activation="sigmoid"),
            Transpose(1, 2))

    def forward(self,
                # [batch_size, frame_langth, mel_bands] input,
                input,
                # [batch_size, sentence_length, charactor_embedding_size]
                keys, values,
                timestep,
                target_mask=None,
                src_mask=None,
                speaker_embedding=None):
        x = self.prenet(input)
        x = self.pe(x, start=timestep)
        x.transpose_(0, 1)
        k = keys.transpose(0, 1)
        v = values.transpose(0, 1)
        length = x.shape[0]
        tgt_mask = torch.tensor([[
            -float("inf") if i > j else 0
            for i in range(length)]
                for j in range(length)]).to(x.device)
        attention_list = []
        for layer in self.transformer_decoder:
            x, attention = layer(
                    query=x,
                    keys=k,
                    values=v,
                    tgt_mask=tgt_mask,
                    mem_mask=None,
                    tgt_key_padding_mask=None
                        if target_mask is None else target_mask.squeeze(2),
                    mem_key_padding_mask=None
                        if src_mask is None else src_mask.squeeze(2)
                    )
            attention_list.append(attention)
        x = x.transpose(0, 1).transpose(1, 2)
        # Mel Output
        x = self.postnet(x) + x

        mel = self.conv_mel(x)

        x.transpose_(1, 2)

        # Done
        done = self.fc_done(x)

        return done, mel, x, attention_list

class DeepVoice3(nn.Module):
    def __init__(self):
        super(DeepVoice3, self).__init__()
        if hp.model_encoder == "transformer":
            self.encoder = TransformerEncoder()
        else:
            self.encoder = Encoder()

        if hp.model_decoder == "transformer":
            self.decoder = TransformerDecoder()
        else:
            self.decoder = Decoder()
        self.converter = Converter()

    def forward(self,
                input,
                init_state=None,
                decoder_input=None,
                frame_mask=None,
                script_mask=None):
        keys, values = self.encoder(input)
        if init_state is not None:  # for evaluation
            buf_x = None
            buf_mel = None
            state = init_state
            try:
                for timestep in trange(0, hp.max_timestep, hp.reduction_factor):
                    done, mel, x, attn = self.decoder(
                            input=state,
                            keys=keys,
                            values=values,
                            timestep=timestep if hp.mode == "generate" else 0)

                    if hp.mode == "generate":
                        state = mel
                    else:
                        state = torch.cat([
                            state, mel[:, -hp.reduction_factor:]
                            ], dim=1)
                    if timestep == 0:
                        buf_x = x
                        buf_mel = mel
                    else:
                        buf_x = torch.cat(
                            [buf_x, x[:, -hp.reduction_factor:]],
                            dim=1)
                        buf_mel = torch.cat(
                            [buf_mel, mel[:, -hp.reduction_factor:]],
                            dim=1)
                    attn = sum(attn) / len(attn)
                    if (done > hp.done_threshold).any():
                        break
                    cuda.empty_cache()
            except(KeyboardInterrupt):
                pass
            mel = state

            # from matplotlib import pyplot as plt
            # mel_ = buf_mel.squeeze(0).numpy()
            # plt.imshow(mel_.T)
            # plt.show()

            x = buf_x
            mag = self.converter(x)
            return mel, mag, done, attn

        elif decoder_input is not None:  # for training
            done, mel, x, attn = self.decoder(
                    input=decoder_input,
                    keys=keys,
                    values=values,
                    timestep=0,
                    target_mask=frame_mask,
                    src_mask=script_mask)
            x = x.contiguous()
            mag = self.converter(x)
            return mel, mag, done, attn
        else:
            assert False
