import argparse
import torch

parser = argparse.ArgumentParser(description='Preprocess the checkpoint.')
parser.add_argument('--ckpt_path', type=str, default=None, help='path to pretrained checkpoint')

args = parser.parse_args()

print("Start converting the pretrained checkpoint...")
ckpt = torch.load(args.ckpt_path, map_location=torch.device('cpu')) ## load classification checkpoint
torch.save(ckpt["model"], "backbone-crossformer-s.pth") ## only model weights are needed
print("Finish converting the checkpoint.")