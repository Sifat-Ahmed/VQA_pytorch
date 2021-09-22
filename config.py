import os
import torch
from bnlp import BengaliGlove
import albumentations as A
from albumentations.pytorch import ToTensorV2


class Config:
    def __init__(self):
        
        self.batch_size = 96
        self.epochs = 500
        
        self.resize = True
        self.image_size = (448, 448)

        self.num_workers = 4
        self.pin_memory = True
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        
        self.learning_rate = 0.001
        self.classification_threshold = 0.75

        
        self.bGlove = BengaliGlove()

        
        ## Dataset
        self.number_of_data = 10000
        
        ## All paths 
        
        self.glove_path = r"/home/workstaion/workspace/potatochips/vqa/bn_glove.39M.300d.txt"
        self.bn_vector_path = r'/home/workstaion/workspace/potatochips/vqa/bangla_word2vec/bangla_word2vec/bnwiki_word2vec.vector'
        self.image_path = r'/home/workstaion/workspace/potatochips/vqa/Dataset/train/yes_no_images'
        self.json_path = r'/home/workstaion/workspace/potatochips/vqa/Dataset/train/VQA_Data/yes_no_data.json'

        self.transform = A.Compose(
            [
                A.Normalize(mean = (0.485, 0.456, 0.406),
                            std = (0.229, 0.224, 0.225)),
                ToTensorV2()

            ]
        )