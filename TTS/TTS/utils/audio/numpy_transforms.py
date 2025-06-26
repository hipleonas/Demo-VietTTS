from io import BytesIO
from typing import Tuple
import librosa
import numpy as np
import scipy
import soundfile as sf
from librosa import magphase, pyin

def build_mel_basis(
    *,
    sample_rate: int = None,
    fft_size: int = None,
    num_mels: int = None,
    mel_fmax: int = None,
    mel_fmin: int = None,
    **kwargs,
) -> np.ndarray:
    """
    Xây dựng ma trận cơ sở Mel
    Để biến đổi từ Hz sang Mel
    mel = 2595⋅log10(1 + f/700)
    f = 700 * (10**(mel/2595) - 1)

    Khi thực hiện biến đổi FFT, ta thu được phổ tuyến tính , tức là mỗi bin FFT sẽ ứng với
    1 tấn số cụ thể : freq[i] = i * sample_rate  / fft_size

    Tuy nhiên, phổ này không phù hợp với thang cảm nhận của tai người. 
    Do đó ta cần chuyển đổi nó sang thang Mel bằng cách áp dụng Mel filter bank.

    Ý tưởng cài mel filter bank
    Bước 1: chọn số lượng filter (num_mels) và kích thước FFT (fft_size)
    Bước 2: Chuyển fmin, fmax sang Mel scale
    Bước 3: chia đều mel scale thành num_mels + 2
    """
    if mel_fmax is not None:
        assert mel_fmax >= mel_fmin
        assert mel_fmax <= sample_rate // 2

    return librosa.filters.mel(
        sr = sample_rate,
        n_mels = num_mels,
        n_fft= fft_size,
        fmin = mel_fmin,
        fmax = mel_fmax
    )

def millisec_to_length(*, 
    frame_length_ms: int = None, 
    frame_shift_ms: int = None, 
    sample_rate: int = None, 
    **kwargs
) -> Tuple[int, int]:
    """
    Chuyển đổi về win_length và hop_length
    Ý tưởng cài đặt:
    sample = tần số lấy mẫu (hz) <=> trong 1s có sample_rate mẫu
    Kích thươc cửa số theo số mẫu = tần số lấy mẫu * độ dài cửa sổ (ms) / 1000
    Độ dài bước nhảy (hop_length) = tần số lấy mẫu * độ dài bước nhảy (ms) / 1000
    """
    assert frame_length_ms % frame_shift_ms == 0 , "Frame length must be a multiple of frame shift"
    
    win_length = int (sample_rate * frame_length_ms / 1000)
    n_hop = frame_length_ms / frame_shift_ms # số luọn bước nhảy trong 1 cửa sổ
    assert n_hop > 0, "Frame shift must be less than frame length"
    hop_length = int(sample_rate / n_hop)
    return win_length, hop_length
  
def amp_to_db(*, x: np.ndarray = None, gain: float = 1, base: int = 10, **kwargs) -> np.ndarray:
    """
    Chuyển đổi từ cường độ âm thanh sang decibel (dB)
    Công thức : db = 10 * log10(x / gain)
    """
    assert (x < 0 ).sum() == 0 ," [!] Input values must be non-negative."
    return gain * _log(np.maximum(1e-8, x), base) 
def db_to_amp(*, x: np.ndarray = None, gain: float = 1, base: int = 10, **kwargs) -> np.ndarray:
    return _exp(x / gain, base)
def preemphasis(*, x: np.ndarray, coef: float = 0.97, **kwargs) -> np.ndarray:
    """
    Thực hiện tiền nhấn (pre-emphasis) trên tín hiệu âm thanh
    Tiền nhấn là quá trình khuếch đại tần số cao của tín hiệu âm thanh
    Công thức: y[n] = x[n] - coef * x[n-1]
    """
    assert  0 <= coef < 1, "Coefficient must be in the range [0, 1)"
    if coef == 0:
        raise RuntimeError(" [!] Preemphasis is set 0.0.")
    return scipy.signal.lfilter([1, -coef], [1], x)

def deemphasis(*, x: np.ndarray = None, coef: float = 0.97, **kwargs) -> np.ndarray:

    assert  0 <= coef < 1, "Coefficient must be in the range [0, 1)"
    if coef == 0:
        raise RuntimeError(" [!] Preemphasis is set 0.0.")
    return scipy.signal.lfilter([1], [1, -coef], x)

def spec_to_mel(*, spec: np.ndarray, mel_basis: np.ndarray = None, **kwargs) -> np.ndarray:
    return mel_basis @ spec

def mel_to_spec(*, mel: np.ndarray = None, mel_basis: np.ndarray = None, **kwargs) -> np.ndarray:
    assert (mel < 0).sum() == 0, " [!] Input values must be non-negative."
    spec = np.linalg.pinv(mel_basis) @ mel
    return np.maximum(spec, 1e-8)  # Tránh giá trị âm hoặc bằng 0 trong phổ

def wav_to_spec(*, wav: np.ndarray = None, **kwargs) -> np.ndarray:
    """Compute a spectrogram from a waveform.

    Args:
        wav (np.ndarray): Waveform. Shape :math:`[T_wav,]`

    Returns:
        np.ndarray: Spectrogram. Shape :math:`[C, T_spec]`. :math:`T_spec == T_wav / hop_length`
    """
    D = stft(y=wav, **kwargs)
    S = np.abs(D)
    return S.astype(np.float32)


def wav_to_mel(*, wav: np.ndarray = None, mel_basis=None, **kwargs) -> np.ndarray:
    """Compute a melspectrogram from a waveform."""
    D = stft(y=wav, **kwargs)
    S = spec_to_mel(spec=np.abs(D), mel_basis=mel_basis, **kwargs)
    return S.astype(np.float32)


def spec_to_wav(*, spec: np.ndarray, power: float = 1.5, **kwargs) -> np.ndarray:
    """Convert a spectrogram to a waveform using Griffi-Lim vocoder."""
    S = spec.copy()
    return griffin_lim(spec=S**power, **kwargs)


def mel_to_wav(*, mel: np.ndarray = None, power: float = 1.5, **kwargs) -> np.ndarray:
    """Convert a melspectrogram to a waveform using Griffi-Lim vocoder."""
    S = mel.copy()
    S = mel_to_spec(mel=S, mel_basis=kwargs["mel_basis"])  # Convert back to linear
    return griffin_lim(spec=S**power, **kwargs)
 

def _log(x, base):
    if base == 10:
        return np.log10(x)
    return np.log(x)
def _exp(x, base):
    if base == 10:
        return np.power(10, x)
    return np.exp(x)

### STFT and ISTFT ###
def stft(
     *,
    y: np.ndarray = None,
    fft_size: int = None,
    hop_length: int = None,
    win_length: int = None,
    pad_mode: str = "reflect",
    window: str = "hann",
    center: bool = True,
    **kwargs,
):
    return librosa.stft(
        y=y,
        n_fft=fft_size,
        hop_length=hop_length,
        win_length=win_length,
        pad_mode=pad_mode,
        window=window,
        center=center,
    )
def istft(
    *,
    y: np.ndarray = None,
    hop_length: int = None,
    win_length: int = None,
    window: str = "hann",
    center: bool = True,
    **kwargs,
) -> np.ndarray:
    """Librosa iSTFT wrapper.

    Check http://librosa.org/doc/main/generated/librosa.istft.html argument details.

    Returns:
        np.ndarray: Complex number array.
    """
    return librosa.istft(y, hop_length=hop_length, win_length=win_length, center=center, window=window)

def griffin_lim(*, spec: np.ndarray = None, num_iter=60, **kwargs) -> np.ndarray:
    return

def compute_stft_paddings(
    *, x: np.ndarray = None, hop_length: int = None, pad_two_sides: bool = False, **kwargs
) -> Tuple[int, int]:
    return

def compute_f0(
    *,
    x: np.ndarray = None,
    pitch_fmax: float = None,
    pitch_fmin: float = None,
    hop_length: int = None,
    win_length: int = None,
    sample_rate: int = None,
    stft_pad_mode: str = "reflect",
    center: bool = True,
    **kwargs,
) -> np.ndarray:    
    
    return

def compute_energy(y: np.ndarray, **kwargs) -> np.ndarray:
    return

###Xử lí âm thanh###
def load_wav(*, filename: str, sample_rate: int = None, resample: bool = False, **kwargs) -> np.ndarray:
    """
    Nạp file âm thanh WAV từ path file name 
    resample: nếu True thì sẽ resample về sample_rate
    """
    if resample:
        x, _ = librosa.load(filename, sr = sample_rate, resample = True)
    else:
        # SF is faster than librosa for loading files
        x, _ = sf.read(filename)
    return x

def save_wav(*, wav: np.ndarray, path: str, sample_rate: int = None, pipe_out=None, **kwargs) -> None:
    BITS16 = 2**15
    wav_norm = wav * (BITS16 /max(0.01, np.max(np.abs(wav))))  
    wav_norm = wav_norm.astype(np.int16)
    if pipe_out:
        wav_buffer = BytesIO()
        scipy.io.wavfile.write(wav_buffer, sample_rate, wav_norm)
        wav_buffer.seek(0)
        pipe_out.write(wav_buffer.read())
    
    scipy.io.wavfile.write(path, sample_rate, wav_norm)

def find_endpoint(
    *,
    wav: np.ndarray = None,
    trim_db: float = -40,
    sample_rate: int = None,
    min_silence_sec=0.8,
    gain: float = None,
    base: int = None,
    **kwargs,
) -> int:
    return

def trim_silence(
    *,
    wav: np.ndarray = None,
    sample_rate: int = None,
    trim_db: float = None,
    win_length: int = None,
    hop_length: int = None,
    **kwargs,
) -> np.ndarray:
    
    return

def volume_norm(*, x: np.ndarray = None, coef: float = 0.95, **kwargs) -> np.ndarray:
    return


def rms_norm(*, wav: np.ndarray = None, db_level: float = -27.0, **kwargs) -> np.ndarray:
    return


def rms_volume_norm(*, x: np.ndarray, db_level: float = -27.0, **kwargs) -> np.ndarray:
    return

def mulaw_encode(*, wav: np.ndarray, mulaw_qc: int, **kwargs) -> np.ndarray:
    """
    Làm trơn tín hiệu (nén logarit) để giữ chi tiết tốt hơn ở biên độ nhỏ – vì tai người nhạy với âm thanh nhỏ.

    f(x) = sign(x) * log(1 + mu * |x|) / log(1 + mu) -1 <= x <= 1
    Trong đó mu là tham số điều chỉnh độ nén, thường là 255.

    Sau đó ta chuyển thành giá trị rời rạc:


    """
    u = np.power(2 , mulaw_qc) - 1
    signal_fx = np.sign(wav) * np.log(1 + u * np.abs(wav)) / np.log(1.0 + u)
    signal_fx = (signal_fx + 1) / 2 * u + 0.5
    return np.floor(signal_fx)

def mulaw_decode(*, wav, mulaw_qc: int, **kwargs) -> np.ndarray:
    """
    Ngược lại với encode thoi
    """
    u = np.power(2, mulaw_qc) - 1
    x = np.sign(wav) * (1/ u) * (np.power(1 + u, np.abs(wav)) - 1) 
    return x

def quantize(*, x: np.ndarray, quantize_bits: int, **kwargs) -> np.ndarray:
    return (x + 1.0) * (2**quantize_bits - 1) / 2


def dequantize(*, x: np.ndarray, quantize_bits: int, **kwargs) -> np.ndarray:
    return 2 * x / (2**quantize_bits - 1) - 1

def encode_16bits(*, x: np.ndarray, **kwargs) -> np.ndarray:
    return np.clip(x * 2**15, -(2**15), 2**15 - 1).astype(np.int16)
