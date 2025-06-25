from dataclasses import dataclass, asdict
from TTS.encoder.configs.base_encoder_config import BaseEncoderConfig

@dataclass
class SpeakerEncoderConfig(BaseEncoderConfig):

    model: str = "speaker_encoder"
    class_name_key : str = "speaker_name"