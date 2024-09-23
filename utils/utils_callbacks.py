import logging
import os
import time
from typing import List

import torch
import random
from numpy.linalg import norm
from eval import verification
from utils.utils_logging import AverageMeter
import numpy as np
from pyeer.eer_info import get_eer_stats
from pyeer.cmc_stats import get_cmc_curve
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm

#Quality import
from utils.quality_utils.edc import EdcErrorType
from utils.quality_utils.edc import EdcSample
from utils.quality_utils.edc import EdcSamplePair
from utils.quality_utils.edc import EdcOutput
from utils.quality_utils.edc import compute_edc
from utils.quality_utils.edc import compute_edc_pauc
from utils.quality_utils.edc import compute_edc_area_under_theoretical_best

comparison_type_to_error_type = {
    'mated': EdcErrorType.FNMR,
    'nonmated': EdcErrorType.FMR,
}


class CallBackEDC(object):
    def __init__(self, frequent, database):
        self.database = database
        self.frequent: int = frequent
        self.best_pauc: float = 1000
        self.best_step = 0
        self.mated_comparisons = []
        self.non_mated_comparisons = []
        self.backbone = None
        

    def __call__(self, num_update, backbone: torch.nn.Module):
        if num_update > 0 and num_update % self.frequent == 0:
            logging.info('Evaluating the backbone')
            backbone.eval()
            self.ver_test(backbone, num_update)
            backbone.train()

    def save_best_performance(self, output):
        torch.save(self.backbone.state_dict(), os.path.join(output, "best_backbone.pth"))
        np.savetxt(os.path.join(output, 'mated_comparisons.txt'), np.array(self.mated_comparisons))
        np.savetxt(os.path.join(output, 'non_mated_comparisons.txt'), np.array(self.non_mated_comparisons))

    def ver_test(self, backbone: torch.nn.Module, global_step: int):
        test_dataloader = DataLoader(self.database, batch_size=32, shuffle=False, num_workers=2)
        dataset = {}
        quality_scores_per_sample = {}
        for _, (img, label, idx) in enumerate(test_dataloader):
            img = img.cuda()
            with torch.no_grad():
                features, qs = backbone(img)
                features = F.normalize(features)

            features = features.detach().cpu().numpy()
            label = label.detach().cpu().numpy()
            qs = qs.detach().cpu().numpy()
            idnx = idx.detach().cpu().numpy()

            for i in range(len(idnx)):
                filename = test_dataloader.dataset.images[idnx[i]]
                quality_scores_per_sample[filename] = qs[i]

                if label[i] in dataset:
                    dataset[label[i]].append((filename, features[i]))
                else:
                    dataset[label[i]] = [(filename, features[i])]
 
        #Perform Verification
        pauc, eer, fmr01 = self.get_quality_assessment(dataset, quality_scores_per_sample)
        
        if pauc < self.best_pauc:
            self.best_pauc = pauc

        del dataset
        del test_dataloader

        logging.info(
                '[%s][%d]Best-PAUC: %1.5f, PAUC: %1.5f, EER: %1.5f, FMR01: %1.5f'  % ('HaGRID', global_step, self.best_pauc, pauc, eer, fmr01))


    @staticmethod
    def perform_verification(dataset):
        keys = list(dataset.keys())
        random.shuffle(keys)
        genuine_list, impostors_list = {}, {}
        #Compute mated comparisons
        for k in keys:
            for i in range(len(dataset[k]) - 1):
                fr, reference = dataset[k][i]
                for j in range(i + 1, len(dataset[k])):
                    fp, probe = dataset[k][j]
                    value = np.dot(probe,reference)/(norm(probe)*norm(reference))
                    genuine_list['{}&{}'.format(fp, fr)] = value
        #Compute non-mated comparisons
        for i in range(len(keys)):
            fr, reference = random.choice(dataset[keys[i]])
            for j in range(len(keys)):
                if i != j:
                    fp, probe = random.choice(dataset[keys[j]])
                    value = np.dot(probe,reference)/(norm(probe)*norm(reference))
                    impostors_list['{}&{}'.format(fp, fr)] = value

        return genuine_list, impostors_list

    @staticmethod
    def _min_max_normalize(quality_scores: dict, bin_count: int) -> dict:
        value_min = min(quality_scores.values())
        value_max = max(quality_scores.values())
        value_range = value_max - value_min
        return {
            sample_id: round(bin_count * ((quality_score - value_min) / value_range))
            for sample_id, quality_score in quality_scores.items()
        }

    @staticmethod
    def get_quality_assessment(dataset:dict, quality_scores_sample:dict, min_max_normalize = 0, comparison_type = 'mated', starting_error = 0.05, pauc_discard_limit = 0.20):
        similarity_scores, non_mated = CallBackEDC.perform_verification(dataset=dataset)
        quality_scores_per_algorithm = {'CR_FIQA': quality_scores_sample}

        edc_outputs = {}
        for quality_assessment_algorithm, quality_scores in tqdm(quality_scores_per_algorithm.items(), desc='EDC curves'):
            # Apply the example min-max normalization if set (deactivated by default):
            if min_max_normalize > 0:
                quality_scores = CallBackEDC._min_max_normalize(quality_scores, min_max_normalize)
            # Prepare sample and sample pair structures for compute_edc:
            samples = {sample_id: EdcSample(quality_score=quality_score) for sample_id, quality_score in quality_scores.items()}
            sample_pairs = []
            for pair_id, similarity_score in similarity_scores.items():
                sample_id1, sample_id2 = pair_id.split('&')
                sample_pairs.append(
                    EdcSamplePair(
                        samples=(
                            samples[sample_id1],
                            samples[sample_id2],
                        ),
                        similarity_score=similarity_score,
                    ))

            # Run compute_edc:
            error_type = comparison_type_to_error_type[comparison_type]
            edc_output = compute_edc(
                error_type=error_type,
                sample_pairs=sample_pairs,
                starting_error=starting_error,
            )
            edc_outputs[quality_assessment_algorithm] = edc_output

        pauc_values = {}
        for quality_assessment_algorithm, edc_output in edc_outputs.items():
            pauc_value = compute_edc_pauc(edc_output, pauc_discard_limit)
            pauc_value -= compute_edc_area_under_theoretical_best(edc_output, pauc_discard_limit)
            pauc_values[quality_assessment_algorithm] = pauc_value
        

        mated = list(similarity_scores.values())
        non_mated = list(non_mated.values())

        stat = get_eer_stats(mated, non_mated)

        return pauc_values['CR_FIQA'], stat.eer*100, stat.fmr1000*100

    


class CallBackVerification(object):
    def __init__(self, frequent, rank, val_targets, rec_prefix, image_size=(112, 112)):
        self.frequent: int = frequent
        self.rank: int = rank
        self.highest_acc: float = 0.0
        self.highest_acc_list: List[float] = [0.0] * len(val_targets)
        self.ver_list: List[object] = []
        self.ver_name_list: List[str] = []
        if self.rank == 0:
            self.init_dataset(val_targets=val_targets, data_dir=rec_prefix, image_size=image_size)

    def ver_test(self, backbone: torch.nn.Module, global_step: int):
        results = []
        for i in range(len(self.ver_list)):
            acc1, std1, acc2, std2, xnorm, embeddings_list = verification.test(
                self.ver_list[i], backbone, 10, 10)
            logging.info('[%s][%d]XNorm: %f' % (self.ver_name_list[i], global_step, xnorm))
            logging.info('[%s][%d]Accuracy-Flip: %1.5f+-%1.5f' % (self.ver_name_list[i], global_step, acc2, std2))
            if acc2 > self.highest_acc_list[i]:
                self.highest_acc_list[i] = acc2
            logging.info(
                '[%s][%d]Accuracy-Highest: %1.5f' % (self.ver_name_list[i], global_step, self.highest_acc_list[i]))
            results.append(acc2)

    def init_dataset(self, val_targets, data_dir, image_size):
        for name in val_targets:
            path = os.path.join(data_dir, name + ".bin")
            if os.path.exists(path):
                data_set = verification.load_bin(path, image_size)
                self.ver_list.append(data_set)
                self.ver_name_list.append(name)

    def __call__(self, num_update, backbone: torch.nn.Module):
        if self.rank == 0 and num_update > 0 and num_update % self.frequent == 0:
            backbone.eval()
            self.ver_test(backbone, num_update)
            backbone.train()


class CallBackLogging(object):
    def __init__(self, frequent, rank, total_step, batch_size, world_size, writer=None, resume=0, rem_total_steps=None):
        self.frequent: int = frequent
        self.rank: int = rank
        self.time_start = time.time()
        self.total_step: int = total_step
        self.batch_size: int = batch_size
        self.world_size: int = world_size
        self.writer = writer
        self.resume = resume
        self.rem_total_steps = rem_total_steps

        self.init = False
        self.tic = 0

    def __call__(self, global_step, loss: AverageMeter, epoch: int, std:float, center:float):
        if self.rank == 0 and global_step > 0 and global_step % self.frequent == 0:
            if self.init:
                try:
                    speed: float = self.frequent * self.batch_size / (time.time() - self.tic)
                    speed_total = speed * self.world_size
                except ZeroDivisionError:
                    speed_total = float('inf')

                time_now = (time.time() - self.time_start) / 3600
                # TODO: resume time_total is not working
                if self.resume:
                    time_total = time_now / ((global_step + 1) / self.rem_total_steps)
                else:
                    time_total = time_now / ((global_step + 1) / self.total_step)
                time_for_end = time_total - time_now
                if self.writer is not None:
                    self.writer.add_scalar('time_for_end', time_for_end, global_step)
                    self.writer.add_scalar('loss', loss.avg, global_step)
                msg = "Speed %.2f samples/sec   Loss %.4f Margin %.4f Center %.4f Epoch: %d   Global Step: %d   Required: %1.f hours" % (
                    speed_total, loss.avg, std, center,epoch, global_step, time_for_end
                )
                logging.info(msg)
                loss.reset()
                self.tic = time.time()
            else:
                self.init = True
                self.tic = time.time()

class CallBackModelCheckpoint(object):
    def __init__(self, rank, output="./"):
        self.rank: int = rank
        self.output: str = output

    def __call__(self, global_step, backbone: torch.nn.Module, header: torch.nn.Module = None):
        if global_step > 100 and self.rank == 0:
            torch.save(backbone.module.state_dict(), os.path.join(self.output, str(global_step)+ "backbone.pth"))
        if global_step > 100 and header is not None:
            torch.save(header.module.state_dict(), os.path.join(self.output, str(global_step)+ "header.pth"))
