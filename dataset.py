import numbers
import os
import queue as Queue
import threading

# import mxnet as mx
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from pathlib import Path
import glob, cv2
from PIL import Image
import pandas as pd
from collections import defaultdict 


class BackgroundGenerator(threading.Thread):
    def __init__(self, generator, local_rank, max_prefetch=6):
        super(BackgroundGenerator, self).__init__()
        self.queue = Queue.Queue(max_prefetch)
        self.generator = generator
        self.local_rank = local_rank
        self.daemon = True
        self.start()

    def run(self):
        torch.cuda.set_device(self.local_rank)
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def next(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self


class DataLoaderX(DataLoader):
    def __init__(self, local_rank, **kwargs):
        super(DataLoaderX, self).__init__(**kwargs)
        self.stream = torch.cuda.Stream(local_rank)
        self.local_rank = local_rank

    def __iter__(self):
        self.iter = super(DataLoaderX, self).__iter__()
        self.iter = BackgroundGenerator(self.iter, self.local_rank)
        self.preload()
        return self

    def preload(self):
        self.batch = next(self.iter, None)
        if self.batch is None:
            return None
        with torch.cuda.stream(self.stream):
            for k in range(len(self.batch)):
                self.batch[k] = self.batch[k].to(device=self.local_rank,
                                                 non_blocking=True)

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        if batch is None:
            raise StopIteration
        self.preload()
        return batch


class HaGrid(Dataset):
    def __init__(self, root_dir, perception_file, hand='right', human_perception=False, transform=None, isTraining = False):
        super(HaGrid, self).__init__()
        self.transform = transform
        self.root_dir = root_dir
        self.perception_file = perception_file
        self.human_perception = human_perception
        self.hand = hand
        self.isTraining = isTraining
        self.num_gestures = 9
        self.identities, self.images, self.id_per_sample, self.targets, self.quality_percetible = self._scan()

        assert len(self.images) == len(self.targets)

        if self.use_masks:
            assert len(self.correct_masks) == len(self.masks) == len(self.images) 

    def _scan(self):
        identities = {}
        labels = []
        images_path = []
        id_per_sample = []
        quality_percetible = []

        if self.human_perception:
            df = pd.read_csv(self.perception_file)
            perception_dict = dict(df.values)
            max_value = max(list(perception_dict.values()))
            min_value = min(list(perception_dict.values()))
            perception_dict = {k: (v - min_value)/(max_value - min_value) for k, v in perception_dict.items()}

        input_path = self.root_dir
        subject_id = list(os.listdir(input_path))

        if self.isTraining:
            stats = defaultdict(dict)
            for idx in subject_id:
                gestures = os.listdir(os.path.join(input_path, idx))

                if '.DS_Store' in gestures:
                    gestures.remove('.DS_Store')
                #loading gestures   
                for g in gestures:
                    input_folder = os.path.join(input_path, idx, g, self.hand)
                    if os.path.exists(input_folder):
                        samples = len(list(Path(input_folder).glob('*.jpg')))
                        if samples > 1:
                            stats[idx][g] = samples

            subject_id = list(filter(lambda k: len(list(k[1].values())) >= self.num_gestures, stats.items()))
            subject_id = [k for k, v in subject_id] 

        id_count = 0
        for id in subject_id:
            images = list(glob.glob(('{}/{}/*/{}/*.jpg'.format(input_path, id, self.hand)), recursive=True))
            if len(images) > 1:
                identities[id] = images.copy()
                labels = [*labels, *[id_count] * len(images)]
                if self.human_perception:
                    quality_percetible = [*quality_percetible, *[float(perception_dict['{}/{}/{}/{}'.format(Path(img).parent.parent.parent.name, Path(img).parent.parent.name, Path(img).parent.name, Path(img).name)]) for img in images]]
                images_path = [*images_path, *images]
                id_per_sample = [*id_per_sample, *[id] * len(images)]
                id_count += 1
                
        return identities, images_path, id_per_sample, labels, quality_percetible


    def number_identities(self):
        return len(set(self.targets))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        img = cv2.imread(self.images[index])
        
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img = self.transform(img)

        return img, torch.tensor((self.targets[index]), dtype=(torch.long)), torch.tensor((self.quality_percetible[index]), dtype=(torch.float)) if self.human_perception else torch.tensor((index), dtype=(torch.long)).item()
