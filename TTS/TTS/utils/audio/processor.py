from io import BytesIO
from typing import Dict, Tuple
import librosa
import numpy as np
import scipy.io.wavfile as wavfile
import scipy.signal as signal
from coqpit import Coqpit

from TTS.tts.utils.helpers import StandardScaler
from TTs.utils.audio.numpy_transforms import (
    amp_to_db,
    build_mel_basis,
    compute_f0,
    db_to_amp,
    deemphasis,
    find_endpoint,
    griffin_lim,
    load_wav,
    mel_to_spec,
    millisec_to_length,
    preemphasis,
    rms_volume_norm,
    spec_to_mel,
    stft,
    trim_silence,
    volume_norm,
)

class AudioProcessor(object):

    def __init__(
        self,
        sample_rate=None,
        resample=False,
        num_mels=None,
        log_func="np.log10",
        min_level_db=None,
        frame_shift_ms=None,
        frame_length_ms=None,
        hop_length=None,
        win_length=None,
        ref_level_db=None,
        fft_size=1024,
        power=None,
        preemphasis=0.0,
        signal_norm=None,
        symmetric_norm=None,
        max_norm=None,
        mel_fmin=None,
        mel_fmax=None,
        pitch_fmax=None,
        pitch_fmin=None,
        spec_gain=20,
        stft_pad_mode="reflect",
        clip_norm=True,
        griffin_lim_iters=None,
        do_trim_silence=False,
        trim_db=60,
        do_sound_norm=False,
        do_amp_to_db_linear=True,
        do_amp_to_db_mel=True,
        do_rms_norm=False,
        db_level=None,
        stats_path=None,
        verbose=True,
        **_,
    ):
        # setup class attributed
        self.sample_rate = sample_rate
        self.resample = resample
        self.num_mels = num_mels
        self.log_func = log_func
        self.min_level_db = min_level_db or 0
        self.frame_shift_ms = frame_shift_ms
        self.frame_length_ms = frame_length_ms
        self.ref_level_db = ref_level_db
        self.fft_size = fft_size
        self.power = power
        self.preemphasis = preemphasis
        self.griffin_lim_iters = griffin_lim_iters
        self.signal_norm = signal_norm
        self.symmetric_norm = symmetric_norm
        self.mel_fmin = mel_fmin or 0
        self.mel_fmax = mel_fmax
        self.pitch_fmin = pitch_fmin
        self.pitch_fmax = pitch_fmax
        self.spec_gain = float(spec_gain)
        self.stft_pad_mode = stft_pad_mode
        self.max_norm = 1.0 if max_norm is None else float(max_norm)
        self.clip_norm = clip_norm
        self.do_trim_silence = do_trim_silence
        self.trim_db = trim_db
        self.do_sound_norm = do_sound_norm
        self.do_amp_to_db_linear = do_amp_to_db_linear
        self.do_amp_to_db_mel = do_amp_to_db_mel
        self.do_rms_norm = do_rms_norm
        self.db_level = db_level
        self.stats_path = stats_path

        """
        Điều chỉnh cơ số
        """
        if self.log_func == "np.log":
            self.base = np.e
        elif self.log_func == "np.log10":
            self.base = 10
        else:
            raise ValueError(
                f"Unsupported log function: {self.log_func}. "
                "Supported functions are: np.log, np.log10"
            )
        
        if hop_length is None:
            self.win_length, self.hop_length = None
        else:
            self.hop_length = hop_length
            self.win_length = win_length


        assert(
            self.win_length <= self.fft_size
        ) , f" [!] win_length cannot be larger than fft_size - {self.win_length} vs {self.fft_size}"
        assert min_level_db is not None or min_level_db  == 0.0, " [!] min_level_db is 0"

        members = vars(self)

        if verbose:
            print('Setting up AudioProcessor with the following parameters:')
            for k, v in members.items():
                print(" | > {}:{}".format(k,v))

        self.mel_basis = None

        if stats_path and signal_norm:
            mel_mean, mel_std, linear_mean, linear_std, _ = self.load_state(stats_path)
            self.setup_scaler(mel_mean, mel_std, linear_mean, linear_std)
            self.signal_norm = True
            self.max_norm = None
            self.clip_norm = None
            self.symmetric_norm = None
    @staticmethod
    def init_from_config(self, config: "Coqpit", verbose = True):
        if "audio" in config:
            return AudioProcessor(verbose = verbose, **config.audio)
        return AudioProcessor(verbose = verbose,**config)
    def normalize(self, S: np.ndarray) -> np.ndarray:

        """
        chuẩn hóa melspectrogram theo trung bình và phương sai
        Kích thước của S: số dòng = số lượng đặc trưng
        Bước 1: Nếu có signal_form 
        Bước 2: kiếm tra tiếp nếu thuộc tính mels_scaler hay không(hasattr(sel.mel_scaler)) ?
        Bước 3: Nếu có thì thực hiện chuẩn hóa
            Chuẩn hóa mel_scaler khi số lượng mel = [0]
            chuẩn hóa linear_scaler khi số lượng fft_size // 2 = mel.shape
        """
        tempS = S.copy()

        if self.signal_norm:
            if hasattr(self, "mel_scaler"):
                n_features = tempS.shape[0]
                if n_features == self.num_mels:
                    return self.mel_scaler.transform(tempS.T).T
                elif n_features == self.fft_size / 2:
                    return self.linear_scaler.transform(tempS.T).T
                else:
                    raise RuntimeError("Mean-Var stats does not match the given feature dimensions.") 
            tempS -= self.ref_level_db
            normS = (tempS - self.min_level_db) / (-self.min_level_db)
            if self.symmetric_norm:
                normS = ((2 * self.max_norm) * normS) - self.max_norm

                if self.clip_norm:
                    normS = np.clip(
                        normS, -self.max_norm, self.max_norm  # pylint: disable=invalid-unary-operand-type
                    )
                return normS
            else:
                normS = self.max_norm * normS
                if self.clip_norm:
                    normS = np.clip(normS, 0, self.max_norm)
                return normS
            
        return tempS
    def denormalize(self, S: np.ndarray) -> np.ndarray:
        """
        Tương tự như hàm normalize hàm denorrmalize cũng có chức năng tương tự mà ngược lại
        """

        tempS = S.copy()
        if self.signal_norm:
            if hasattr(self, "mel_scaler"):
                n_feats = tempS.shape[0]

                if n_feats == self.num_mels:
                    return self.mel_scaler.inverse_transform(tempS.T).T
                if n_feats == self.fft_size / 2:
                    return self.linear_scaler.inverse_transform(tempS.T).T
                raise RuntimeError("Mean-Var stats does not match the given feature dimensions.") 
            
            if self.symmetric_norm:
                if self.clip_norm:
                    tempS = np.clip(tempS, - self.max_norm, self.max_norm)
                tempS = ((tempS + self.max_norm) * -self.min_level_db / (2 * self.max_norm)) + self.min_level_db
                tempS += self.ref_level_db
                return tempS
            else:
                if self.clip_norm:
                    tempS = np.clip(tempS, 0, self.max_norm)
                tempS = (tempS * -self.min_level_db / self.max_norm) + self.min_level_db
                tempS += self.ref_level_db
                return tempS
        return tempS


    def load_state(self, state_path : str) -> Tuple[np.array, np.array, np.array, np.array, Dict]:
        statistic_params = np.load(state_path, allow_pickle = True).item()

        mel_mean, mel_std = statistic_params["mel_mean"], statistic_params["mel_std"]
        linear_mean, linear_std = statistic_params["linear_mean"], statistic_params["linear_std"]
        audio_config = statistic_params["audio_config"]

        skip_params = ["griffin_lim_iters", "stats_path", "do_trim_silence", "ref_level_db", "power"]
        """
        Các thông số griffin, state_path, do_trim_silence, ref_level_db, power KHÔNG ảnh hưởng đến việc tính toán mean-var stats
        """
        for param in audio_config:
            if param in skip_params:
                continue
            if param not in ["sample_rate","trim_db"]:
                assert(
                    audio_config[param] == self.__dict__[param]
                ),f"Fatal:Audio param {param} does not match the value used for computing mean-var stats. {audio_config[param]} vs {self.__dict__[param]} "
        return mel_mean, mel_std, linear_mean, linear_std, audio_config
    
    def setup_scaler(
        self,
        mel_mean: np.ndarray,
        mel_std : np.ndarray,
        linear_mean: np.ndarray,   
        linear_std: np.ndarray,
    ) -> None:
        self.mel_mean = StandardScaler()
        self.linear_mean = StandardScaler()

        self.mel_scaler.set_state(mel_mean, mel_std)
        self.linear_scaler.set_state(linear_mean, linear_std)


    def apply_preemphasis(self, x: np.ndarray) -> np.ndarray:
        return preemphasis(x=x, coef=self.preemphasis)

    def apply_inv_preemphasis(self, x: np.ndarray) -> np.ndarray:
        return deemphasis(x=x, coef=self.preemphasis)

    """
    Xử lí Spectrogram
    """

    def spectrogram(self, y: np.ndarray) -> np.ndarray:
        if self.preemphasis != 0:
            y = self.apply_preemphasis(y)
        D = stft(
            y=y,
            fft_size=self.fft_size,
            hop_length=self.hop_length,
            win_length=self.win_length,
            pad_mode=self.stft_pad_mode,
        )
        if self.do_amp_to_db_linear:
            S = amp_to_db(x=np.abs(D), gain=self.spec_gain, base=self.base)
        else:
            S = np.abs(D)
        return self.normalize(S).astype(np.float32)
    
    def melspectrogram(self, y: np.ndarray) -> np.ndarray:
        """Compute a melspectrogram from a waveform."""
        if self.preemphasis != 0:
            y = self.apply_preemphasis(y)
        D = stft(
            y=y,
            fft_size=self.fft_size,
            hop_length=self.hop_length,
            win_length=self.win_length,
            pad_mode=self.stft_pad_mode,
        )
        S = spec_to_mel(spec=np.abs(D), mel_basis=self.mel_basis)
        if self.do_amp_to_db_mel:
            S = amp_to_db(x=S, gain=self.spec_gain, base=self.base)

        return self.normalize(S).astype(np.float32)
    def inv_spectrogram(self, spectrogram: np.ndarray) -> np.ndarray:
        """Convert a spectrogram to a waveform using Griffi-Lim vocoder."""
        S = self.denormalize(spectrogram)
        S = db_to_amp(x=S, gain=self.spec_gain, base=self.base)
        # Reconstruct phase
        W = self._griffin_lim(S**self.power)
        return self.apply_inv_preemphasis(W) if self.preemphasis != 0 else W

    def inv_melspectrogram(self, mel_spectrogram: np.ndarray) -> np.ndarray:
        """Convert a melspectrogram to a waveform using Griffi-Lim vocoder."""
        D = self.denormalize(mel_spectrogram)
        S = db_to_amp(x=D, gain=self.spec_gain, base=self.base)
        S = mel_to_spec(mel=S, mel_basis=self.mel_basis)  # Convert back to linear
        W = self._griffin_lim(S**self.power)
        return self.apply_inv_preemphasis(W) if self.preemphasis != 0 else W

    def out_linear_to_mel(self, linear_spec: np.ndarray) -> np.ndarray:
        """Convert a full scale linear spectrogram output of a network to a melspectrogram.

        Args:
            linear_spec (np.ndarray): Normalized full scale linear spectrogram.

        Returns:
            np.ndarray: Normalized melspectrogram.
        """
        S = self.denormalize(linear_spec)
        S = db_to_amp(x=S, gain=self.spec_gain, base=self.base)
        S = spec_to_mel(spec=np.abs(S), mel_basis=self.mel_basis)
        S = amp_to_db(x=S, gain=self.spec_gain, base=self.base)
        mel = self.normalize(S)
        return mel
    def _griffin_lim(self, S):
        return griffin_lim(
            spec=S,
            num_iter=self.griffin_lim_iters,
            hop_length=self.hop_length,
            win_length=self.win_length,
            fft_size=self.fft_size,
            pad_mode=self.stft_pad_mode,
        )
    def compute_f0(self, x: np.ndarray) -> np.ndarray:

        if len(x) % self.hop_length == 0:
            x = np.pad(x, (0, self.hop_length // 2), mode=self.stft_pad_mode)

        f0 = compute_f0(
            x=x,
            pitch_fmax=self.pitch_fmax,
            pitch_fmin=self.pitch_fmin,
            hop_length=self.hop_length,
            win_length=self.win_length,
            sample_rate=self.sample_rate,
            stft_pad_mode=self.stft_pad_mode,
            center=True,
        )

        return f0
    def find_endpoint(self, wav: np.ndarray, min_silence_sec=0.8) -> int:

        return find_endpoint(
            wav=wav,
            trim_db=self.trim_db,
            sample_rate=self.sample_rate,
            min_silence_sec=min_silence_sec,
            gain=self.spec_gain,
            base=self.base,
        )
    def trim_silence(self, wav):   
        return trim_silence(
            wav = wav,
            sample_rate=self.sample_rate,
            trim_db=self.trim_db,
            win_length=self.win_length,
            hop_length=self.hop_length,
        )
    @staticmethod
    def sound_norm(x: np.ndarray) -> np.ndarray:
        return volume_norm(x=x)

    def rms_volume_norm(self, x: np.ndarray, db_level: float = None) -> np.ndarray:
        if db_level is None:
            db_level = self.db_level
        return rms_volume_norm(x=x, db_level=db_level)
    ### save and load ###
    def load_wav(self, filename: str, sr: int = None) -> np.ndarray:

        """
        Đọc file wav từ thử viện lỉbrosa
        filename là tên file
        sr = samplerate là tấn số lấy mẫu

        ý tưởng code:
        Nếu sr tham số != None thì lấy sr của tham số
        ko thì lấy của hàm
        """
        x = None
        if sr is not None:
            x = load_wav(filename = filename, sample_rate = sr, resample = True)
        else:
            x = load_wav(filename = filename, sample_rate = self.sample_rate, resample = self.resample)

        if self.do_trim_silence:
            try:
                x = self.trim_silence(x)
            except ValueError:
                print(f" [!] File cannot be trimmed for silence - {filename}")        
        if self.do_sound_norm:
            try: 
                x = self.sound_norm(x)
            except ValueError:
                print(f" [!] File cannot be sound normalize - {filename}")        
        if self.do_rms_norm:
            try:
                x = self.rms_volume_norm(x, self.db_level)
            except ValueError:
                print(f" [!] File cannot be rm volume norm - {filename}")        
        return x
    def save_wav(self, wav: np.ndarray, path: str, sr: int = None, pipe_out=None) -> None:
        """Save a waveform to a file using Scipy.

        Args:
            wav (np.ndarray): Waveform to save.
            path (str): Path to a output file.
            sr (int, optional): Sampling rate used for saving to the file. Defaults to None.
            pipe_out (BytesIO, optional): Flag to stdout the generated TTS wav file for shell pipe.
        """
        if self.do_rms_norm:
            wav_form = self.rms_volume_norm(wav, self.db_level) * 32676
        else:
            wav_norm = wav * (32767 / max(0.01, np.max(np.abs(wav))))

        wav_norm = wav_norm.astype(np.int16)

        if pipe_out :
            wav_buffer = BytesIO()
            wavfile.write(wav_buffer, sr if sr else self.sample_rate, wav_norm)
            wav_buffer.seek(0)
            pipe_out.buffer.write(wav_buffer.read())
        wavfile.write(path, sr if sr else self.sample_rate, wav_norm)


    def get_duration(self, filename: str) -> float:
        return librosa.get_duration(filename = filename)