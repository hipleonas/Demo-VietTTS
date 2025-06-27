import base64
import collections
import os
import random
from typing import Dict, List, Union

import numpy as np
import torch
import tqdm
from torch.utils.data import Dataset

from TTS.tts.utils.data import prepare_data, prepare_stop_target, prepare_tensor
from TTS.utils.audio import AudioProcessor
from TTS.utils.audio.numpy_transforms import compute_energy as calculate_energy

import mutagen