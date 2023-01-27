#IMPORT LIBRARIES
import sys
from pathlib import Path
import glob
import os
from tqdm import tqdm
from shutil import copyfile
import pickle

import numpy as np
import cv2

from scipy import ndimage
import skimage
from skimage.transform import resize
from skimage.morphology import label
from skimage.color import rgb2gray
from sklearn.model_selection import train_test_split
from skimage.morphology import remove_small_holes, remove_small_objects, label
from skimage.feature import peak_local_max

from skimage.segmentation import watershed

sys.path.append('..')
sys.path.append('../dataset_loader')
sys.path.append('../model')

from dataset_loader.image_loader import *
from model.resunet import *
from utils import *
from torch.utils.data import DataLoader

from config import *

import torch
from torchvision.transforms import transforms as T
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib
import argparse
from kneed import KneeLocator
matplotlib.use('qt5Agg')
import shutil


from evaluation.evaluation_utils import post_processing, compute_metrics_global, model_inference, compute_metrics

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

def main(learning):

    for l in learning:
        learning = '../model_results/{}/'.format(l)
        files = os.listdir(learning)
        if files[0] == 'green':
            path = learning + 'green/'
            model_names = os.listdir(path)
            for mn in model_names:
                path_to_remove = [path + mn + '/summary', path + mn + '/metrics', path +mn]
                for prm in path_to_remove:
                    shutil.rmtree(prm, ignore_errors=True)
                    print('remove', prm)
                    os.makedirs(path +mn, exist_ok=True)
        else:
            learning = learning + files[0]
            files = os.listdir(learning)
            if files[0] == 'green':
                path = learning + '/green/'
                model_names = os.listdir(path)
                for mn in model_names:
                    path_to_remove = [path + mn + '/summary', path + mn + '/metrics', path+mn]
                    for prm in path_to_remove:
                        shutil.rmtree(prm, ignore_errors=True)
                        print('remove', prm)
                        os.makedirs(path+mn, exist_ok=True)


if __name__ == "__main__":

    learning = ['few_shot', 'fine_tuning']
    main(learning)
