from datasets.thicknessDataset import ThicknessDataset, get_transforms
import lightning.pytorch as pl
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np


class ThicknessDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, train_df=None, val_df=None, test_df=None, classification=False, batch_size=32, image_size=256):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transforms_train, self.transforms_val = get_transforms(image_size, None, None)

        class_weights = {}
        for c in np.unique(train_df['multiclass_thick_label']):
            class_weights[c] = 1 / len(train_df[train_df['multiclass_thick_label'] == c])
            print(c, len(train_df[train_df['multiclass_thick_label'] == c]), 1 / len(train_df[train_df['multiclass_thick_label'] == c]))
        # class_weights = {0: 1 / len(train_df[train_df['stage_ajcc'] != 6]), 1: 1 / len(train_df[train_df['stage_ajcc'] == 6])}

        self.valid_dataset = ThicknessDataset(val_df,
                                              img_path=self.data_dir,
                                              transforms=self.transforms_val,
                                              classification=classification,
                                              ext='.jpg',
                                              test=False
                                              )
        self.train_dataset = ThicknessDataset(train_df,
                                              img_path=self.data_dir,
                                              transforms=self.transforms_train,
                                              classification=classification,
                                              ext='.jpg',
                                              test=False
                                              )
        if test_df is not None:
            self.test_dataset = ThicknessDataset(test_df,
                                                 img_path=r'C:\Users\Aleksandra\PycharmProjects\DermaAnalysis\data\ISIC2018_Task3_Test_Input\ISIC2018_Task3_Test_Input',
                                                 classification=classification,
                                                 ext='.jpg',
                                                 transforms=self.transforms_val,
                                                 test=True
                                                 )

        self.weighted_sampler = WeightedRandomSampler(
            weights=[class_weights[i] for i in train_df.multiclass_thick_label],
            num_samples=len(train_df),
            replacement=True
        )
        print([class_weights[i] for i in train_df.multiclass_thick_label])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, drop_last=True,
                          shuffle=False, num_workers=3, persistent_workers=True, sampler=self.weighted_sampler
                          )

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, num_workers=3, persistent_workers=True)
