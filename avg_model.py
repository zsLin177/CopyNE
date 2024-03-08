import argparse
import os
import torch

def avg(src_path, dst_model):
    # get model names from src_path
    src_model_paths = [os.path.join(src_path, f) for f in os.listdir(src_path) if f.endswith('.model') and not f.startswith('current') and not 'avg' in f]
    dst_model_path = os.path.join(src_path, dst_model)
    num = len(src_model_paths)
    avg = None
    for path in src_model_paths:
        print(f'Processing {path}')
        states = torch.load(path, map_location=torch.device('cpu'))
        if "model_state_dict" in states:
            states = states["model_state_dict"]
        if avg is None:
            avg = states
        else:
            for k in avg.keys():
                avg[k] += states[k]
    # average
    for k in avg.keys():
        if avg[k] is not None:
            # pytorch 1.6 use true_divide instead of /=
            avg[k] = torch.true_divide(avg[k], num)
    print(f'Saving to {dst_model_path}')
    torch.save(avg, dst_model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='average model')
    parser.add_argument('--dst_model', required=True, help='averaged model')
    parser.add_argument('--src_dir',
                        required=True,
                        help='dir contains src models for average')
    args = parser.parse_args()

    avg(args.src_dir, args.dst_model)

    