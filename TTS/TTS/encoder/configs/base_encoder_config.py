from dataclasses import asdict, dataclass, field
from typing import Dict, List
from coqpit import MISSING

from TTS.config.shared_configs import BaseAudioConfig, BaseDatasetConfig, BaseTrainingConfig


@dataclass
class BaseEncoderConfig(BaseTrainingConfig):

    '''
    Các thông số của field cấu hình gồm có
        + model: str
        + 
    '''
    model : str = None
    audio : BaseAudioConfig = field(default_factory= BaseAudioConfig)
    datasets: List[BaseDatasetConfig] = field(default_factory= lambda : [BaseDatasetConfig()])

    '''
    Tham số mô hình
    '''

    model_params: Dict = field(
        #Sử dụng cấu hình mặc định là lstm
        default_factory = lambda : {
            "model_name":"lstm",
            "input_dim": 80, 
            "proj_dim": 256,
            "lstm_dim": 768,
            "num_lstm_layers": 3,
            "use_lstm_with_projection": True,
        }
    )

    #Cấu hình tăng cường âm thahh
    audio_augmentation : Dict = field (
        default_factory= lambda :{

        }
    )

    '''
    Tham số huấn luyện
    '''
    epochs: int = 10000
    lr_decay: bool = False
    warmup_steps: int = 4000
    loss : str = "angleproto"
    grad_clip: float = 3.0
    lr: float = 1e-4
    optimizer: str = "radam"
    optimizer_params: Dict = field(default_factory= lambda :\
        {"betas": [0.9, 0.999], "weight_decay": 0}                   
    )
    '''
    Tham số dataloader
    '''
    num_loader_workers:int = MISSING
    voice_len: float = 1.6
    #1 class = 1 speaker id
    #1 speaker id có thể chứa nhiều utterance
    num_classes_in_batch: int = MISSING
    num_utter_per_class: int = MISSING
    eval_num_classes_in_batch: int = None
    eval_num_utter_per_class: int = None


    '''
    Tham số tensorboard
    '''
    tb_model_param_stats: bool = False
    steps_plot_stats: int = 10
    save_step: int = 1000
    print_step: int = 20
    run_eval: bool = False

    def check_values(self):
        super().check_values()
        c = asdict(self)
        assert (
            c["model_params"]["input_dim"] == self.audio.num_mels
        ), " [!] model input dimendion must be equal to melspectrogram dimension."
