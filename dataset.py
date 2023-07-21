import os
import pandas as pd
from random import seed, shuffle
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

class ManualDS(Dataset):
    def __init__(self,
                 comp_ds_num: int,
                 data_root: str,
                 ds_names: list,
                 random_state: int) -> None:
        '''
        comp_ds_num: int.
            [0-6]. The index of reserved dataset to alter the dataset distribution.
        data_root: str.
            The directory contains folders that are named after each dataset's name.
            Each folder should contain a subfolder and a '.csv' file.
            The subfolder should be named 'imgs' with gene images,
            and the name of the '.csv' file with 'pos' and 'neg' two columns
            should be '<ds_name>_labels.csv'.
        ds_names: A list of the names of subfolders contained in 'data_root'.
        '''

        super().__init__()
        self.data_root = data_root
        self.ds_names = ds_names
        # self.ds_names = [
        #                 "Adult_Mouse_Kidney_FFPE",
        #                 "Human_Breast_Cancer_Ductal_Carcinoma_In_Situ_Invasive_Carcinoma_FFPE",
        #                 "Human_Colorectal_Cancer_Whole_Transcriptome_Analysis",
        #                 "Human_Glioblastoma_Whole_Transcriptome_Analysis",
        #                 "Human_Ovarian_Cancer_Whole_Transcriptome_Analysis_Stains_DAPI_Anti-PanCK_Anti-CD45",
        #                 "Human_Prostate_Cancer_Adjacent_Normal_Section_with_IF_staining_FFPE",
        #                 "Human_Spinal_Cord_Whole_Transcriptome_Analysis_Stains_DAPI_Anti-SNAP25_Anti-GFAP_Anti-Myelin_CNPase"]

        self.ds_path_list = [os.path.join(self.data_root, i)
                             for i in self.ds_names]


        self.comp_ds_num = comp_ds_num

        self.ds_paths = []

        self.init_labels(random_state)
        self.change_mode(mode = "train")

    def __getitem__(self, index):
        if self.mode in ["train","test"]:
            img_path = self.ds_paths[index][0]
            label = self.ds_paths[index][1]
        else:
            img_path = self.ds_paths[index]
        gene_name = img_path.split("/")[-1].split(".jpg")[0]

        with Image.open(img_path)as img:
                data = self.transforms(img)        
        if self.mode in ["train","test"]:
            return gene_name, data, label
        else:
            return gene_name, data

    def __len__(self):
        return len(self.ds_paths)

    def init_labels(self, random_state):
        self.labels = [pd.read_csv(os.path.join(self.data_root, i,
                     f"{i}_labels.csv"), header = 0, index_col = 0,
                     sep = ",")
                     for i in self.ds_names]
        self.pos_labels = []
        self.neg_labels = []
        for i in range(len(self.labels)):
            self.pos_labels.append([[os.path.join(self.ds_path_list[i],"imgs",f"{j}.jpg"), 1]
                                for j in self.labels[i]["pos"]])
            self.neg_labels.append([[os.path.join(self.ds_path_list[i],"imgs",f"{j}.jpg"), 0]
                                for j in self.labels[i]["neg"]])


        self.ds_all = [[os.path.join(i, "imgs", j) for j in os.listdir(i + "/imgs")]
                        for i in self.ds_path_list]

        self.seg_ds = []
        
        for i in range(len(self.ds_names)):
            if i == self.comp_ds_num:
                continue
            self.seg_ds.extend(self.pos_labels[i])
            self.seg_ds.extend(self.neg_labels[i])
        
        seed(random_state)
        shuffle(self.seg_ds)

    def change_mode(self, mode = "test", leave = 0):
        """
        mode: str.
            'train', 'test' or 'comp'.
            If 'train' or 'test', this dataset module will contain
            5/6 or 1/6 (leave-one-out) labeled images that are mixed
            and shuffled from 5 datasets, excluding the 'self.comp_ds_num'-th
            dataset in self.ds_names.
            Else, this dataset module will contain all labeled images for the
            'self.comp_ds_num'-th dataset in self.ds_names.
        leave: int.
            [0-5]. Parameter of 'leave-one-out'.
        """
                
        self.mode = mode
        
        if leave < 0 or leave > len(self.ds_names)-1:
            raise ValueError("'leave' should be integer and " +\
                             "greater than or equal to 0 " +\
                             f"and less than or equal to {len(self.ds_path_list)}")
        self.reset_data()
        self.set_tranforms()

        if self.mode not in ['train', 'test', 'comp']:
            raise ValueError("'mode' should be one of ['train', 'test', 'comp']")

        seg_len = int(len(self.seg_ds)/6)
        if self.mode == "train":
            for seg in range(len(self.seg_ds)-1):
                if seg == leave:
                    continue
                seg_ds = self.seg_ds[(seg * seg_len):((seg + 1) * seg_len)]
                self.ds_paths.extend(seg_ds)

        elif self.mode == "test":
            seg_ds = self.seg_ds[(leave * seg_len):((leave + 1) * seg_len)]
            self.ds_paths.extend(seg_ds)

        else: # self.mode == "comp"
            self.ds_paths.extend(self.ds_all[self.comp_ds_num])


    def set_tranforms(self):
        if self.mode == "train":
            self.transforms = T.Compose([T.RandomRotation((0, 360)),
                                         T.RandomHorizontalFlip(),
                                         T.Resize([128, 128]),
                                         T.ToTensor()])
        else:
            self.transforms = T.Compose([T.Resize([128, 128]),
                                         T.ToTensor()])

    def reset_data(self):
        self.ds_paths = []

class Pred_DS(Dataset):
    def __init__(self, dir_path: str) -> None:
        """
        dir_path: str.
            The directory of gene imgs that generated for a dataset by draw.py
        """
        super().__init__()
        imgs_list = os.listdir(dir_path)
        self.paths = [os.path.join(dir_path, i) for i in imgs_list]
        self.gene_names = [img.split(".jpg")[0] for img in imgs_list]
        self.transforms = T.Compose([T.Resize([128, 128]),
                                         T.ToTensor()])
    def __getitem__(self, index):
        with Image.open(self.paths[index]) as img:
            data = self.transforms(img)
        return self.gene_names[index], data
    def __len__(self):
        return len(self.paths)

