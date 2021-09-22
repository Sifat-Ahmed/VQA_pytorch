import torchtext
from torchtext.legacy.data import Field, TabularDataset, BucketIterator
from torchtext.vocab import Vectors, GloVe
import helper.preprocessing
import cv2, os


class LoadDataset:
    def __init__(self, cfg, text, image, label, mode) -> None:

        self._cfg = cfg
        self._text = text
        self._image = image
        self._label = label
        self._mode = mode.tolower()
        
        assert len(self._text) == len(self._image) and len(self._image) == len(self._label), "number of text, image and labels are not same"
        assert self._mode == "train" or self._mode == "test", "The value of mode is not correct (train/test)"
        
    def __len__(self):
        return len(self._image)
    
    def __getitem__(self, index):
        text = self._text[index]
        label = self._label[index]
    
        img_path = "COCO_%s2014_%s.jpg" % (self._mode, self._image[index].zfill(12))
        image = cv2.imread(os.path.join(self._cfg.image_path, img_path))
        
        if self._cfg.resize:
            image = cv2.resize(image, self._cfg.image_size)

        if self._cfg.transform:
            image = self._cfg.transform(image=image)["image"]

        return text, image, label
        
            
        
    