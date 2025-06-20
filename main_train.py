import json
import torch
import torch.backends.cudnn as cudnn
from utils import set_seed, create_dir
import lightning.pytorch as pl
from thicknessClassifier import ThicknessClassifier
from thicknessDataModule import ThicknessDataModule
from lightning.pytorch.tuner.tuning import Tuner
import pandas as pd
from lightning.pytorch import loggers as pl_loggers


with open('config.json') as config_file:
    paths = json.load(config_file)

save_dir = paths['savePath']
create_dir(save_dir)

cudnn.benchmark = True

seeds = [68616]
cfg = {
    "batch_size": 64,
    'img_size': 224,
    'optim': 'sgd',  # not used
    'lr': 0.0004867242410956716,
    'momentum': 0.9,
    'weight_decay': 1e-3,
    "classification": True,
    "dropout": 0.5,
    "outputs": 3,
    'net_name': 'convnext_tiny.in12k_ft_in1k',
    'gamma': 0.981
}

if __name__ == '__main__':
    for seed in seeds:
        torch.set_float32_matmul_precision('high')
        set_seed(seed)
        df = pd.read_csv(paths['csvPath'])
        for split in range(2, 5):
            print("split", split)
            df_train = df[df['split'] != split]
            df_valid = df[df['split'] == split]

            model = ThicknessClassifier(True, classification=cfg['classification'], backbone=cfg['net_name'],
                                        dropout=cfg['dropout'], out=cfg['outputs'],
                                        lr=cfg['lr'],
                                        wd=cfg['weight_decay'], momentum=cfg['momentum'])

            data_module = ThicknessDataModule(
                data_dir=paths['imagesPath'],
                train_df=df_train,
                val_df=df_valid,
                image_size=cfg['img_size'],
                batch_size=cfg['batch_size'],
                classification=cfg['classification']
            )

            callbacks = [
                # pl.callbacks.BatchSizeFinder(),
                pl.callbacks.ModelCheckpoint(
                    dirpath=save_dir,
                    filename='ckpt_best_{validation_loss:02f}-{epoch:02d}-{validation_metric:02f}'
                             f'_{cfg["net_name"]}_'
                             f'_lr={cfg["lr"]}'
                             f'_mom={cfg["momentum"]}'
                             f'_wd={cfg["weight_decay"]}'
                             f'_out={cfg["outputs"]}'
                             f'_bs={cfg["batch_size"]}'
                             f'_imgSize={cfg["img_size"]}'
                             f'_{"class"if cfg["classification"] else "reg"}'
                             f'_size={cfg["img_size"]}_{seed}_{split}',
                    monitor='validation_metric',
                    mode='max',
                    save_on_train_epoch_end=True,
                    save_top_k=1
                ),
                pl.callbacks.ModelCheckpoint(
                    dirpath=save_dir,
                    filename='ckpt_best_{validation_loss:02f}-{epoch:02d}-{validation_metric:02f}'
                             f'_{cfg["net_name"]}_'
                             f'_lr={cfg["lr"]}'
                             f'_mom={cfg["momentum"]}'
                             f'_wd={cfg["weight_decay"]}'
                             f'_out={cfg["outputs"]}'
                             f'_bs={cfg["batch_size"]}'
                             f'_imgSize={cfg["img_size"]}'
                             f'_{"class" if cfg["classification"] else "reg"}'
                             f'_size={cfg["img_size"]}_{seed}_{split}',
                    monitor='validation_loss',
                    mode='min',
                    save_on_train_epoch_end=True,
                    save_top_k=1
                ),

                pl.callbacks.ModelCheckpoint(
                    dirpath=save_dir,
                    filename='ckpt_{validation_loss:02f}-{epoch:02d}-{validation_metric:02f}'
                             f'_{cfg["net_name"]}_'
                             f'_lr={cfg["lr"]}'
                             f'_mom={cfg["momentum"]}'
                             f'_wd={cfg["weight_decay"]}'
                             f'_out={cfg["outputs"]}'
                             f'_bs={cfg["batch_size"]}'
                             f'_imgSize={cfg["img_size"]}'
                             f'_{"class"if cfg["classification"] else "reg"}'
                             f'_size={cfg["img_size"]}_{seed}_{split}',
                    save_on_train_epoch_end=True,
                    every_n_epochs=1,
                    save_last=True
                ),
                pl.callbacks.LearningRateMonitor(logging_interval='epoch'),
                # pl.callbacks.LearningRateFinder(early_stop_threshold=None)
            ]

            tb_logger = pl_loggers.TensorBoardLogger(save_dir="logs/",
                                                     version=f'{cfg["net_name"]}'
                                                             f'_lr={cfg["lr"]}'
                                                             f'_mom={cfg["momentum"]}'
                                                             f'_wd={cfg["weight_decay"]}'
                                                             f'_out={cfg["outputs"]}'
                                                             f'_bs={cfg["batch_size"]}'
                                                             f'_imgSize={cfg["img_size"]}'
                                                             f'_{"class"if cfg["classification"] else "reg"}'
                                                             f'_size={cfg["img_size"]}_{seed}_{split}')
            trainer = pl.Trainer(callbacks=callbacks, accelerator='gpu', devices=1, max_epochs=200, logger=tb_logger,
                                 log_every_n_steps=1, default_root_dir=save_dir)
            tuner = Tuner(trainer)

            # Fit model
            trainer.fit(model=model, datamodule=data_module)

            model.freeze()
            trainer.validate(model=model, datamodule=data_module)

            trainer.save_checkpoint("final.ckpt")
