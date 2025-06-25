from dataclasses import dataclass, asdict

from TTS.encoder.configs.base_encoder_config import BaseEncoderConfig

@dataclass
class EmotionEncoderConfig(BaseEncoderConfig):
    model : str = "emotion_encoder"
    class_name_by_key : str = "emotion_name"
    map_classid_to_classname: dict = None

# if __name__ == '__main__':

#     emotion_config = EmotionEncoderConfig(
#         model = "emo_encoder",
#         class_name_by_key = "emotion_label",
#         map_classid_to_classname={
#             0: "neutral",
#             1: "happy", 
#             2: "sad",
#             3: "angry",
#             4: "surprised"
#         }
#     )

#         # Khi sử dụng trong training
#     print(emotion_config.model)  # Output: "emo_encoder_v2"
#     print(emotion_config.map_classid_to_classname[1])  # Output: "happy"