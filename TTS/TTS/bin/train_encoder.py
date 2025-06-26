import os
import sys
import traceback
import time
import torch


torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enable = True
torch.manual_seed(54321)



cuda_usage = torch.cuda.is_available()
n_gpus = torch.cuda.device_count()

print("Cuda available:", cuda_usage )
print("Number of GPUs:", n_gpus)

def setup_loader(ap:AudioProcessor, is_val:bool = True, verbose: bool = False):
    pass

def main(args):
    pass
if __name__ == "__main__":

    args ,  config, OUT_PATH , AUDIO_PATH, config_logger, dashboard_logger  = init_training()
    try:
        main(args)

    except KeyboardInterrupt as keyinterupt:
        remove_experiment_folder(OUT_PATH)
        try:
            exit(0)
        except SystemExit:
            os._exit(0)
    except Exception as e:
        print('Exception occurred:', e )
        traceback.print_exc()
        remove_experiment_folder(OUT_PATH)
        sys.exit(1)


