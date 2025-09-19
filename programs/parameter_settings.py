import argparse
import hashlib
import os
from datetime import datetime

def parameter_settings(params, job_name, sim_index):
    args = argparse.Namespace(verbose=False, verbose_1=False)
    for k, v in params.items():
        setattr(args, k, v)
    
    print("Model Settings:")
    for key, value in vars(args).items():
        print(f"{key} --> {value}")
        
    now = datetime.now()
    args.timestamp = now.strftime("%Y_%m_%d_%H_%M_%S")  # Use underscores instead of hyphens
    args.filename = f"exp_{args.timestamp}_{job_name}_sim_{sim_index}"
    args.results_dir = f"models/{args.experiment}/{args.filename}"
    os.makedirs(args.results_dir, exist_ok=True)

    return args 