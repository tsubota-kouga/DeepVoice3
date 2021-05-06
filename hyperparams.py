
class HyperParams:
    debug = False

    mode = "train"  # train, generate
    dp = 0.05
    use_dropout = True
    use_norm = None  # "batch"
    batch_size = 16
    num_epoch = 1000

    done_threshold = 0.2
    done_out_of_range = 0.0

    train_update_data_per = 30
    eval_update_data_per = 5

    init_distribution = "xavier"

    activation = "swish"

    pad_kind = "constant"

    eps = 1e-8
    lr = 0.0005
    weight_decay = 0
    use_ams_grad = True

    mel_bce_rate = 0.1
    mag_bce_rate = 0.1
    entropy_rate = 0  # 1e-7
    max_gradient_norm = 100
    max_gradient_value = 5

    max_timestep = 1000
    max_sentence_length = 512

    mel_bands = 80
    reduction_factor = 4
    speaker_size = 1

    corpus = "ljspeech"
    lang = "en"

    if lang == "en":
        id_to_vocab = list("\0abcdefghijklmnopqrstuvwxyz ,.?!'")
        vocab_to_id = {k: i for (i, k) in enumerate(id_to_vocab)}
        vocab_size = len(id_to_vocab)
    elif lang == "ja":
        id_to_vocab = list("\0abcdefghijklmnopqrstuvwxyz ,.?!'あいうえおかきくけこさしすせそなにぬねのはひふへほまみむめもやゆよらりるれろわをん")
        vocab_to_id = {k: i for (i, k) in enumerate(id_to_vocab)}
        vocab_size = len(id_to_vocab)
    else:
        assert("unsupported language")

    preemphasis_coef = 0.97
    sample_rate = 22050
    # fft
    fft_window_kind = "hann"
    fft_size = 2048
    fft_window_size = fft_size
    fft_window_shift = fft_size // 4
    # dB
    top_db = 100.0
    ref_db = 20.0

    # transfomer
    transformer_encoder_d_model = 256
    transformer_decoder_d_model = 512
    transformer_encoder_dim_feedforward = 512
    transformer_decoder_dim_feedforward = 1024
    transformer_num_head_encoder = 1
    transformer_num_head_decoder = 2

    # encoder
    transformer_encoder_prenet_layers = 4
    transformer_encoder_postnet_layers = 6
    transformer_encoder_layers = 1
    transformer_decoder_layers = 2

    model_encoder = "DeepVoice3"

    if model_encoder == "transformer":
        charactor_embedding_size = transformer_encoder_d_model
        encoder_conv_width = 3
        encoder_norm = "batch"
        encoder_dilation_base = 3
    else:
        charactor_embedding_size = 256
        encoder_conv_width = 3
        encoder_norm = "weight"
        encoder_dilation_base = 3
        encoder_layer_size = 10
        encoder_conv_channels = 256

    model_decoder = "DeepVoice3"

    if model_decoder == "transformer":
        decoder_norm = "weight"
        decoder_pre_dilation_base = 3
        decoder_prenet_affine_size = 256
        decoder_prenet_layer_size = 2
        decoder_prenet_conv_width = 3
        decoder_post_dilation_base = 1
        decoder_postnet_layer_size = 6
        decoder_postnet_conv_width = 3
        decoder_dp = 0.1
        converter_input_channel_size = transformer_decoder_d_model

        guided_attention_g = 0.2
    else:
        decoder_pre_dilation_base = 2
        decoder_dilation_base = 1
        decoder_post_dilation_base = 1
        decoder_norm = "weight"
        decoder_prenet_affine_size = 128
        decoder_postnet_affine_size = 512

        decoder_prenet_layer_size = 3
        decoder_layer_size = 2
        decoder_conv_layer_size = [3, 3]
        assert decoder_layer_size == len(decoder_conv_layer_size)
        decoder_postnet_layer_size = 5
        decoder_dp = 0.05

        decoder_prenet_conv_width = 3
        decoder_conv_width = 3
        decoder_postnet_conv_width = 3

        attention_num_head = 1
        guided_attention_g = 0.2
        converter_input_channel_size = decoder_postnet_affine_size

    # attention
    use_guided = num_epoch
    attention_dp = dp
    attention_hidden_size = 128
    position_encoding_auto = False
    position_encoding_scaled = True
    # converter
    converter_dilation_base = 1
    upsample_rate = 4
    converter_pool_kind = "avg"
    converter_pool_kernel_size = 8
    converter_norm = "weight"
    converter_pre_layer_size = 4
    converter_layer_size = 4
    converter_post_layer_size = 4
    converter_pre_conv_channels = 512
    converter_conv_channels = 512
    converter_post_conv_channels = 512
    converter_conv_width = 3
    converter_conv_trans_width = 5
    converter_dp = 0.04

    # for multi speaker
    speaker_embedding_size = 16

    # for single speaker
    w_query = 1.0
    w_key = max_timestep / max_sentence_length


