import json
import torch
import torch.backends.cudnn as cudnn
from utils import set_seed, create_dir
import lightning.pytorch as pl
from thicknessClassifier import ThicknessClassifier
from thicknessDataModule import ThicknessDataModule
import pandas as pd
from lightning.pytorch import loggers as pl_loggers
import optuna
from optuna.integration import PyTorchLightningPruningCallback

with open('config.json') as config_file:
    paths = json.load(config_file)

save_dir = paths['savePath']
create_dir(save_dir)

cudnn.benchmark = True

seed = 68616

cfg = {
    "batch_size": 64,
    'img_size': 224,
    'optim': 'sgd',  # not used
    'lr': 0.001,
    'momentum': 0.9,
    'weight_decay': 1e-5,
    "classification": True,
    "dropout": 0.2,
    "outputs": 3,
    'net_name': 'convnext_tiny.in12k_ft_in1k',
}
df = pd.read_csv(r"C:\Users\Aleksandra\PycharmProjects\thicknessPrediction\train_classification1.csv")

df_train = df[df['split'] != 1]
df_valid = df[df['split'] == 1]


def objective(trial):
    gamma = trial.suggest_float("gamma", 0.98, 1, step=0.001)
    e = trial.suggest_int("e", -3, -1)
    dropout = trial.suggest_float("dropout", 0.5, 0.9, step=0.1)
    lr = trial.suggest_float("lr", 1e-6, 0.001)
    momentum = 0.9
    wd = 10 ** e
    torch.set_float32_matmul_precision('high')

    set_seed(seed)
    model = ThicknessClassifier(True, classification=cfg['classification'], backbone=cfg['net_name'],
                                dropout=dropout, out=cfg['outputs'],
                                lr=lr, gamma=gamma,
                                wd=wd, momentum=cfg['momentum'], focal_gamma=2)

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
                     f'_lr={lr}'
                     f'_mom={momentum}'
                     f'_wd={10 ** e}'
                     f'_out={cfg["outputs"]}'
                     f'_bs={cfg["batch_size"]}'
                     f'_imgSize={cfg["img_size"]}'
                     f'_{"class" if cfg["classification"] else "reg"}'
                     f'_size={cfg["img_size"]}_{seed}',
            monitor='validation_loss',
            mode='min',
            save_on_train_epoch_end=True,
            save_top_k=1
        ),

        pl.callbacks.ModelCheckpoint(
            dirpath=save_dir,
            filename='ckpt_{validation_loss:02f}-{epoch:02d}-{validation_metric:02f}'
                     f'_{cfg["net_name"]}_'
                     f'_lr={lr}'
                     f'_mom={momentum}'
                     f'_wd={10 ** e}'
                     f'_out={cfg["outputs"]}'
                     f'_bs={cfg["batch_size"]}'
                     f'_imgSize={cfg["img_size"]}'
                     f'_{"class" if cfg["classification"] else "reg"}'
                     f'_size={cfg["img_size"]}_{seed}',
            save_on_train_epoch_end=True,
            every_n_epochs=1,
            save_last=True
        ),
        pl.callbacks.LearningRateMonitor(logging_interval='epoch'),
        PyTorchLightningPruningCallback(trial, monitor="validation_metric")
        # pl.callbacks.LearningRateFinder(early_stop_threshold=None)
    ]

    tb_logger = pl_loggers.TensorBoardLogger(save_dir="logs/",
                                             version=f'{cfg["net_name"]}'
                                                     f'_lr={lr}'
                                                     f'_mom={momentum}'
                                                     f'_wd={10 ** e}'
                                                     f'_out={cfg["outputs"]}'
                                                     f'_bs={cfg["batch_size"]}'
                                                     f'_imgSize={cfg["img_size"]}'
                                                     f'_{"class" if cfg["classification"] else "reg"}'
                                                     f'_size={cfg["img_size"]}_{seed}')
    trainer = pl.Trainer(callbacks=callbacks, accelerator='gpu', devices=1, max_epochs=100, logger=tb_logger,
                         log_every_n_steps=1, default_root_dir=save_dir)
    hyperparameters = dict(wd=10 ** e, lr=lr, gamma=gamma, dropout=dropout)
    trainer.logger.log_hyperparams(hyperparameters)

    # Fit model
    trainer.fit(model=model, datamodule=data_module)

    return trainer.callback_metrics["best_validation_metric"].item()


if __name__ == "__main__":
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=5, n_startup_trials=2)

    study = optuna.create_study(storage="sqlite:///db.sqlite3", study_name="thickness_multiclass_convnext_focal_v2",
                                direction="maximize", pruner=pruner, load_if_exists=True)
    study.optimize(objective, n_trials=100, timeout=None)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
