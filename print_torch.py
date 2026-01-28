import sys, os 
sys.path.append(os.path.expanduser("~")+"/dotfiles")
import myutils
import torch

myutils.print_torch(torch)
