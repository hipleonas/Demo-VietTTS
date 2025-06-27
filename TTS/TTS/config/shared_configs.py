from dataclasses import dataclass, asdict
from typing import List
from coqpit import Coqpit, check_argument
from trainer import TrainerConfig

"""
File cấu hình chung baoo gồm
- BaseAudioConfig: Cấu hình âm thanh cơ bản
- BaseDatasetConfig: Cấu hình tập dữ liệu cơ bản
- BaseTrainingConfig: Cấu hình huấn luyện cơ bản
"""
@dataclass
class BaseAudioConfig(Coqpit):
    """Tham số biến biến đổi fourier ngắn hạn"""
    fft_size        : int = 1024
    win_length      : int = 1024
    hop_length      : int = 256
    frame_shift_ms  : int = None #Bước nhảy 1 frame theo ms
    frame_length_ms : int = None # Kích thước 1 frame theo ms
    stft_pad_mode   : str = "reflect"
    """Tham số xử lí 1 audio"""
    sample_rate     : int = 22050
    do_sound_norm   : bool = False
    log_func        : str = "np.log10"
    ref_level_db    : float = 20.0
    preemphasis     : float = 0.0 # Hệ số tiền nhấn thường là 0.97->0.98
    resample        : bool = False
    """Tham số xử lí silence"""
    do_trim_silence: bool = True
    trim_db: int = 45

    """Chuẩn hóa rms"""
    do_rms_norm: bool = False
    db_level: float = None
    """Tham số xử lí spectrogram"""
    power: float = 1.5
    griffin_lim_iters: int = 60
    """Tham số xử lí mel-spectrogram"""
    num_mels: int = 80
    mel_fmin: float = 0.0
    mel_fmax: float = None
    spec_gain: int = 20
    do_amp_to_db_linear: bool = True
    do_amp_to_db_mel: bool = True
    # f0 params
    pitch_fmax: float = 640.0
    pitch_fmin: float = 1.0
    # normalization params
    signal_norm: bool = True
    min_level_db: int = -100
    symmetric_norm: bool = True
    max_norm: float = 4.0
    clip_norm: bool = True
    stats_path: str = None

    def check_values(self,):

        """Kiểm tra các giá trị của cấu hình"""
        config = asdict(self)
        check_argument("num_mels",
            config, restricted=True, 
            min_val = 10, max_val = 2056
        )
        check_argument("sample_rate" , 
            config, restricted= True, 
            min_val = 512, max_val = 100000
        )
        check_argument("fft_size",     
            config, restricted = True, 
            min_val = 128, max_val = 4058
        )
        check_argument("frame_length_ms", 
            config, restricted = True, 
            min_val = 10, max_val = 1000, aleternative = "win_length"
        )


        check_argument("frame_shift_ms", config, restricted=True, min_val=1, max_val=1000, alternative="hop_length")
        check_argument("preemphasis", config, restricted=True, min_val=0, max_val=1)
        check_argument("min_level_db", config, restricted=True, min_val=-1000, max_val=10)
        check_argument("ref_level_db", config, restricted=True, min_val=0, max_val=1000)
        check_argument("power", config, restricted=True, min_val=1, max_val=5)
        check_argument("griffin_lim_iters", config, restricted=True, min_val=10, max_val=1000)

        # normalization parameters
        check_argument("signal_norm", config, restricted=True)
        check_argument("symmetric_norm", config, restricted=True)
        check_argument("max_norm",config, restricted=True, min_val=0.1, max_val=1000)
        check_argument("clip_norm", config, restricted=True)
        check_argument("mel_fmin", config, restricted=True, min_val=0.0, max_val=1000)
        check_argument("mel_fmax", config, restricted=True, min_val=500.0, allow_none=True)
        check_argument("spec_gain", config, restricted=True, min_val=1, max_val=100)
        check_argument("do_trim_silence", config, restricted=True)
        check_argument("trim_db", config, restricted=True)

        return
@dataclass 
class BaseDatasetConfig(Coqpit):
    formatter : str = ""
    dataset_name: str = ""
    path : str = ""
    meta_file_train: str = ""
    meta_file_val : str = ""
    meta_file_attn_mask: str = ""
    phonemizer: str = ""
    language: str = ""
    ignored_speakers: List[str] = None

    def check_values(self):
        config = asdict(self)

        check_argument("formatter", config, restricted=True)
        check_argument("path", config, restricted=True)
        check_argument("meta_file_train", config, restricted=True)
        check_argument("meta_file_val", config, restricted=False)
        check_argument("meta_file_attn_mask", config, restricted=False)

@dataclass
class BaseTrainingConfig(TrainerConfig):
    model: str = None
    # dataloading
    num_loader_workers: int = 0
    num_eval_loader_workers: int = 0
    use_noise_augment: bool = False

