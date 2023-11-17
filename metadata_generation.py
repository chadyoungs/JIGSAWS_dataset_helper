#! /usr/bin/env python3
'''
Created on Mon Jul 20 17:29:20 2020
@author: xiaoxiaoyang
Update on Wed Nov 1 10:52:00 2023
'''
from collections import defaultdict, OrderedDict
import numpy as np
import os
import pickle
import json

from constant import ROOT, TASK


class MetaData(object):
    def __init__(self, score_grades=2, surgeme_simplified=False, res_file_loc=None):
        self.task = TASK
        self.meta_file_loc = os.path.join(ROOT, TASK, f"meta_file_{TASK}.txt")
        self.trans_file_dir = os.path.join(ROOT, TASK, "transcriptions")
        self.traintestsplit_dir = os.path.join(ROOT, "Experimental_setup", TASK, "unBalanced")
        self.traintestsplit_skill_dir = os.path.join(self.traintestsplit_dir, "SkillDetection")
        self.traintestsplit_classifi_dir = os.path.join(self.traintestsplit_dir, "GestureClassification")
        
        self.score_grades = score_grades
        self.surgeme_simplified = surgeme_simplified

        self.metadata_res = defaultdict(OrderedDict)
        self.train_test_split_res = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
        self.surgeme_dict = dict()

        self.res_file_loc = res_file_loc

        self.get_surgeme()

    def _skill_grade(self, score):
        """
        0, novice
        1, intermediate
        2, expert
        """
        if self.score_grades == 2:
            if self.task == "Knot_Tying":
                if score <= 15:
                    return 0
                else:
                    return 2
            elif self.task == "Suturing":
                if score <= 19:
                    return 0 # novice
                else:
                    return 2 # expert
                    
            elif self.task == "Needle_Passing":
                if score <= 15:
                    return 0
                else:
                    return 2
    
    def _surgeme_simplified(self, raw_surgeme):
        pass

    def _find_surgeme_video(self, trial_name, surgeme_start_frame_idx):
        search_data = self.metadata_res[trial_name]["surgeme_start_end"]
        for idx, (x, y, z) in enumerate(zip(search_data["start_frame_idx"], search_data["end_frame_idx"], search_data["surgeme"])):
            if x == surgeme_start_frame_idx:
                surgeme_video_name = "_".join([trial_name, z, str(idx)]) + ".avi"
                label = self.metadata_res["metadata"]["surgeme_label_mapping"][z]
                return f"{z}/{surgeme_video_name} {y-x} {label}", surgeme_video_name, z

    def get_score(self):
        content = np.loadtxt(self.meta_file_loc, dtype=str)
        for row in content:
            trial_name, trial_score = row[0], int(row[2])
            trial_score_grade = self._skill_grade(trial_score)
            self.metadata_res[trial_name]["score"] = trial_score
            self.metadata_res[trial_name]["grade"] = trial_score_grade

    def train_test_split_skill_detection(self):
        """
        return trial name, score, grade list
        """
        outmethod_list = os.listdir(self.traintestsplit_skill_dir)

        # notice that 'OneTrialOut' not involved in the consideration
        outmethod_list.remove("OneTrialOut")

        for outmethod in outmethod_list:
            outmethod_abs = os.path.join(self.traintestsplit_skill_dir, outmethod)
            out_list = os.listdir(outmethod_abs)
            for out in out_list:
                out_abs = os.path.join(outmethod_abs, out)
                for option in ["Train", "Test"]:
                    out_traintest_abs = os.path.join(out_abs, "itr_1", f"{option}.txt")
                    content = np.loadtxt(out_traintest_abs, dtype=str)
                    
                    content_list = []
                    for row in content:
                        trial_name = "_".join(row[0].split("_")[:2])
                        try:
                            trial_label = self._skill_grade(int(row[1]))
                        except:
                            print(out_abs)
                            
                        content_list.append((trial_name, trial_label))
                    self.train_test_split_res["SkillDetection"][outmethod][out][option] = content_list
    
    def train_test_split_converter(self):
        """
        following with UCF-101's style
        label index count from zero !
        """
        save_traintest_split_dir = os.path.join(ROOT, TASK, "surgeme_classifi_traintestsplit")
        os.makedirs(save_traintest_split_dir, exist_ok=True)

        outmethod_list = os.listdir(self.traintestsplit_classifi_dir)

        # notice that 'OneTrialOut' not involved in the consideration
        outmethod_list.remove("OneTrialOut")

        for outmethod in outmethod_list:
            outmethod_abs = os.path.join(self.traintestsplit_classifi_dir, outmethod)
            out_list = os.listdir(outmethod_abs)
            for out in out_list:
                out_abs = os.path.join(outmethod_abs, out)
                for option in ["Train", "Test"]:
                    out_traintest_abs = os.path.join(out_abs, "itr_1", f"{option}.txt")
                    content = np.loadtxt(out_traintest_abs, dtype=str)
                    
                    save_file_loc = os.path.join(save_traintest_split_dir, f"{outmethod}_{out}_{option}.txt")
                    with open(save_file_loc, "w") as f:
                        for row in content:
                            trial_name = "_".join(row[0].split("_")[:2])
                            surgeme_start_frame_idx = int(row[0].split("_")[2])
                            res_str_list, _, _ = self._find_surgeme_video(trial_name, surgeme_start_frame_idx)

                            f.write(res_str_list + "\n")
    
    def generate_annotation_json(self):
        """
        following with UCF-101's style,
        label index count from zero !

        For PyTorchCon3D repo.
        """
        train_test_mapping = {"Train": "training",
                              "Test": "validation"}
        save_annotation_dir = os.path.join(ROOT, TASK, "surgeme_annotation")
        os.makedirs(save_annotation_dir, exist_ok=True)

        outmethod_list = os.listdir(self.traintestsplit_classifi_dir)

        # notice that 'OneTrialOut' not involved in the consideration
        outmethod_list.remove("OneTrialOut")

        for outmethod in outmethod_list:
            outmethod_abs = os.path.join(self.traintestsplit_classifi_dir, outmethod)
            out_list = os.listdir(outmethod_abs)
            for out in out_list:
                out_abs = os.path.join(outmethod_abs, out)

                annotations = dict()
                annotations["database"], labels_dict = {}, OrderedDict()
                save_file_loc = os.path.join(save_annotation_dir, f"{outmethod}_{out}.json")
                for option in ["Train", "Test"]:
                    out_traintest_abs = os.path.join(out_abs, "itr_1", f"{option}.txt")
                    content = np.loadtxt(out_traintest_abs, dtype=str)
                    for row in content:
                        trial_name = "_".join(row[0].split("_")[:2])
                        surgeme_start_frame_idx = int(row[0].split("_")[2])
                        # label means idx while real_label means surgeme name e.g. 'G1'
                        _, surgeme_video_name, real_label = self._find_surgeme_video(trial_name, surgeme_start_frame_idx)

                        annotations["database"][surgeme_video_name] = {"subset": train_test_mapping[option],
                                                                       "annotations": {"label": real_label}
                                                                    }
                        labels_dict[real_label] = 0
                
                annotations["labels"] = sorted(list(labels_dict.keys()), key=lambda x: int(x[1:]))
                with open(save_file_loc, 'w') as f:
                    json.dump(annotations, f, indent=4)

    def get_surgeme(self):
        """
        return a list of [start_frame_idx, end_frame_idx, surgeme]
        """
        txt_files = [os.path.join(self.trans_file_dir, txt_file) for txt_file in os.listdir(self.trans_file_dir)]
        
        for txt_file in txt_files:
            trial_name = os.path.splitext(os.path.basename(txt_file))[0]
            start_frame_idx_list, end_frame_idx_list, surgeme_list = [], [], []
            content = np.loadtxt(txt_file, dtype=str)
            for row in content:
                start_frame_idx, end_frame_idx, surgeme = int(row[0]), int(row[1]), row[2]
                start_frame_idx_list.append(start_frame_idx)
                end_frame_idx_list.append(end_frame_idx)
                surgeme_list.append(surgeme)

                if surgeme not in self.surgeme_dict:
                    self.surgeme_dict[surgeme] = 1

            
            self.metadata_res[trial_name]["surgery_start_end"] = {"start_frame": int(content[0][0]),
                                                                  "end_frame": int(content[-1][1])}
            self.metadata_res[trial_name]["surgeme_start_end"] = {"start_frame_idx": start_frame_idx_list,
                                                                  "end_frame_idx": end_frame_idx_list,
                                                                  "surgeme": surgeme_list}
        
        self.metadata_res["metadata"] = {"surgeme_list": sorted(list(self.surgeme_dict.keys()), key=lambda x: int(x[1:]))}

        self.metadata_res["metadata"]["surgeme_label_mapping"] = {}
        for idx, surgeme in enumerate(self.metadata_res["metadata"]["surgeme_list"]):
            self.metadata_res["metadata"]["surgeme_label_mapping"][surgeme] = idx
    
    def generate_metadata(self):
        self.get_score()
        self.train_test_split_skill_detection()
    
    def generate_train_test_files(self):
        self.train_test_split_converter()

    def save_to_pkl(self):
        with open(self.res_file_loc, "w") as f:
            pickle.dump(self.metadata_res, f)


if __name__ == "__main__":
    trigger = MetaData()
    #trigger.generate_metadata()
    #trigger.generate_train_test_files()
    #trigger.train_test_split_converter()
    trigger.generate_annotation_json()
