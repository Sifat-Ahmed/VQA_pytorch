import os
import json

import config


class ParseJson:
    def __init__(self, cfg):
        #self._json_path = json_path
        self._cfg = cfg
        
    def parse_json(self, encoding = 'utf-8-sig'):
        
        if not os.path.isfile(self._cfg.json_path):
            raise("JSON file not found")

        self._json_dataset = None
                
        try:
            with open(self._cfg.json_path, encoding = encoding, mode="r") as F:
                self._json_dataset = json.load(F)['result_yes_no_1']
                
            self._json_dataset = self._json_dataset[:self._cfg.number_of_data]
        
        except Exception as e:
            print("Error occured with json file", e)
        
        if self._json_dataset:
            self._answers = list()
            self._questions = list()
            self._images = list()

            for a in self._json_dataset:

                self._answers.append(a['Answer.Column1'])
                self._questions.append(a['Questions In Bangla'])
                self._images.append(str(a['Answer.image_id']))


            return self._questions, self._images, self._answers


if __name__ == "__main__":
    cfg = config.Config()
    jp = ParseJson(cfg)

    q, i, a = jp.parse_json(r'/home/workstaion/workspace/potatochips/vqa/Dataset/train/VQA_Data/yes_no_data.json')

    print(q[:5])
    print(a[:5])
