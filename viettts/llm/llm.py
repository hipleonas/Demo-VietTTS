
"""
Tìm hiểu 1 số paper:https://arxiv.org/pdf/2301.02111
SpeechT5: Unified-Modal Encoder-Decoder Pre-Training for Spoken Language Processing
🔗 Microsoft, 2022
https://arxiv.org/abs/2110.07205

Mô hình encoder-decoder Transformer dùng cho nhiều task: TTS, ASR, speaker ID, v.v.

Liên quan đến:

text_encoder: tách riêng module để dùng chung nhiều task.

spk_embed_dim, spk_embed_affine_layer: đưa thông tin speaker vào hệ thống.

Modular hóa kiến trúc như trong TransformerLM.

EnCodec: High Fidelity Neural Audio Compression
🔗 Meta AI, 2022
https://arxiv.org/abs/2210.13438

Cung cấp discrete token đại diện cho âm thanh – được dùng trong VALL-E.

Liên quan đến:

speech_token_size: kích thước vocabulary token của speech.

Sử dụng discrete representation của speech để sinh bằng LLM.

T5: Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer
🔗 Google, 2020
https://arxiv.org/abs/1910.10683

Thiết kế LLM encoder-decoder với các task khác nhau dưới dạng “text-to-text”.

Liên quan đến:

task_id, llm_embedding: nhúng ID task để dùng mô hình đa nhiệm (multitask).

Kiến trúc modular giống TransformerLM.

5. FastSpeech 2: Fast and High-Quality End-to-End Text to Speech
🔗 Microsoft, 2021
https://arxiv.org/abs/2006.04558

Dùng Transformer-based architecture và speaker embedding để sinh speech.

Liên quan đến:

spk_embed_dim: nhúng thông tin speaker.

length_normalized_loss: giúp training ổn định hơn trên sequence có độ dài khác nhau.
"""
from typing import Dict, Callable, List, Optional, Generator

import torch

import torch.nn as nn
import math
import torch.nn.functional as F

from torch.nn.utils.rnn import pad_sequence, unpad_sequence
from viettts.utils.common import IGNORE_ID
from viettts.transformer.label_smoothing_loss import LabelSmoothingLoss
from viettts.utils.common import th_accuracy

class TransformerLM(nn.Module):

    def __init__(
        self,
        #Bộ mã hóa
        text_encoder : nn.Module,
        #Này <=> với vocab_size
        text_token_dim: int,
        #Hidden size của khi đưa text_token dim vào embedding
        text_encoder_dim: int,
        #Transfomer LLM
        llm: nn.Module,
        llm_input_dim:int,
        llm_output_dim: int,
        #Speech
        speech_token_dim : int,
        sampling: Callable,
        speaker_embed_dim: int = 192,
        lsm_weight: float = 0.0,
        length_norm_loss: bool = False,
        

    ):
        
        super().__init__()
        #TEXT
        self.text_encoder = text_encoder
        self.text_embedding = nn.Embedding(text_token_dim, text_encoder_dim)
        self.text_encoder_affine_layer = nn.Linear(
            self.text_encoder.output_dim(),
            llm_input_dim
        )

        self.llm_input_dim = llm_input_dim
        self.speech_token_size = speech_token_dim
        # TEXT  -> LLM -> Speech
        self.task_id = 1
        self.start_end_os = 0
        self.llm = llm
        self.llm_embedding = nn.Embedding(2, llm_input_dim)
        self.llm_decoder = nn.Linear(llm_output_dim, speech_token_dim + 1)
        #Speech

        self.speech_embedding = nn.Embedding(speech_token_dim, llm_input_dim)
        self.speaker_embed_dim = nn.Linear(
            speaker_embed_dim, llm_input_dim
        )
        #Sampling
        self.sampling = sampling

        #Criterion
        self.criterion_ce = LabelSmoothingLoss(
            size=speech_token_dim + 1,
            padding_idx=IGNORE_ID,
            smoothing=lsm_weight,
            normalize_length=length_norm_loss,
        )
    def encode(
        self, 
        text: torch.Tensor,
        text_len: torch.Tensor,

    ):
        encoder_out, encoder_mask = self.text_encoder(text, text_len, decoding_chunk_size=1, num_decoding_left_chunks=-1)
        encoder_out_len = encoder_mask.squeeze(1).sum(1)
        encoder_out = self.text_encoder_affine_layer(encoder_out)
        return encoder_out, encoder_out_len
    def pad_unpad_sequence(
        self,
        sos_eos_emb,
        embedding,
        text_token,
        text_token_len,
        task_id_emb,
        speech_token,
        speech_token_len
    ):
        text_token = unpad_sequence(text_token, text_token_len.cpu(), batch_first=True)
        speech_token = unpad_sequence(speech_token, speech_token_len.cpu(), batch_first=True)
        lm_input = [torch.concat([sos_eos_emb.squeeze(dim=0), embedding[i], text_token[i], task_id_emb.squeeze(dim=0), speech_token[i]], dim=0)
                    for i in range(len(text_token))]
        lm_input_len = torch.tensor([i.size(0) for i in lm_input], dtype=torch.int32)
        lm_input = pad_sequence(lm_input, batch_first=True, padding_value=IGNORE_ID)
        return lm_input, lm_input_len
    
    @torch.inference_mode()
    def inference(
        self,
        text : torch.Tensor,
        text_len: torch.Tensor,
        prompt_text: torch.Tensor,
        prompt_text_len: torch.Tensor,
        prompt_speech_token: torch.Tensor,
        prompt_speech_token_len: torch.Tensor,
        embedding: torch.Tensor,
        sampling: int = 25,
        max_token_text_ratio: float = 20,
        min_token_text_ratio: float = 2,
    ) -> Generator[torch.Tensor, None, None]:
        """
        Ý tưởng của inference
        Sinh từng token giọng nói (speech token) dựa trên text và 

        Văn bản bản mẫu : prompt_text + giọng nói mẫu (promtp_speech_token)
        embedding : người nói
        ràng buộc độ dài token
        Bước 1: nhận văn bản đầu vào
        Văn bản đầu vào sẽ nối thêm phần gợi ý prompt ở đầu
        """ 
        device = text.device
        text = torch.concat([prompt_text , text], dim = 1)
        text_len  = text_len + prompt_text_len
        text = self.text_embedding(text)
        
        #1. Mã hóa văn bản
        text, text_len = self.encoder(text, text_len)

        #2. Encode embedding

        if embedding.shape[0] != 0:
            embedding = F.normalize(embedding, dim = 1)
            
    def forward(
        self,
        batch: dict,
        device: torch.device
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            text : batch_size x text_len x text_token_dim
            text_len : batch_size
            audio: batch_size x audio_len x speech_token_dim or batch_size x audio_len
            audio_len = batch_size
        """