from pathlib import Path
from typing import Dict
import tomllib

from mads_datasets.base import BaseDatastreamer
from src import models, metrics

import ray
import torch
from filelock import FileLock
from loguru import logger
from mltrainer import (ReportTypes, Trainer, TrainerSettings,
                       rnn_models)
from mltrainer.preprocessors import PaddedPreprocessor, BasePreprocessor
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.search.hyperopt import HyperOptSearch

NUM_SAMPLES = 5
MAX_EPOCHS = 5
DATA_SET = 'ptb'


def train(config: Dict):
    """
    The train function should receive a config file, which is a Dict
    ray will modify the values inside the config before it is passed to the train
    function.
    """
    from src import datasets, metrics
    

    data_dir = Path('../data')
    configfile = Path("config.toml")

    with configfile.open('rb') as f:
        config = tomllib.load(f)
    print(config)

    trainfile = data_dir / (config[DATA_SET] + '_train.parq')
    testfile = data_dir / (config[DATA_SET] + '_test.parq')
    shape = (16, 12)

    traindataset = datasets.HeartDataset2D(trainfile, target="target", shape=shape)
    testdataset = datasets.HeartDataset2D(testfile, target="target", shape=shape)
    

    with FileLock(data_dir / ".lock"):
        # we lock the datadir to avoid parallel instances trying to
        # access the datadir
        trainstreamer = BaseDatastreamer(traindataset, preprocessor=BasePreprocessor(), batchsize=32)
        teststreamer = BaseDatastreamer(testdataset, preprocessor=BasePreprocessor(), batchsize=32)

    # we set up the metric
    # and create the model with the config
    accuracy = metrics.Accuracy()
    recall = metrics.Recall('macro')
    
    modelconfig = models.CNNSettings(**config)
    model = models.ConvBlocks(modelconfig)


    trainersettings = TrainerSettings(
        epochs=MAX_EPOCHS,
        metrics=[accuracy, recall],
        logdir="logs/heart2D",
        train_steps=len(trainstreamer) // 5,  # type: ignore
        valid_steps=len(teststreamer) // 5,  # type: ignore
        reporttypes=[ReportTypes.RAY, ReportTypes.TENSORBOARD, ReportTypes.MLFLOW],
        scheduler_kwargs={"factor": 0.5, "patience": 5},
        earlystop_kwargs=None,
    )

    # because we set reporttypes=[ReportTypes.RAY]
    # the trainloop wont try to report back to tensorboard,
    # but will report back with ray
    # this way, ray will know whats going on,
    # and can start/pause/stop a loop.
    # This is why we set earlystop_kwargs=None, because we
    # are handing over this control to ray.

    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        device = "cpu"  # type: ignore
    logger.info(f"Using {device}")
    if device != "cpu":
        logger.warning(
            f"using acceleration with {device}.Check if it actually speeds up!"
        )

    trainer = Trainer(
        model=model,
        settings=trainersettings,
        loss_fn=torch.nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam,  # type: ignore
        traindataloader=trainstreamer.stream(),
        validdataloader=teststreamer.stream(),
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
        device=str(device),
    )

    trainer.loop()


if __name__ == "__main__":
    ray.init()

    tune_dir = Path("logs/ray").resolve()

    # this example uses BOTH a search algorithm AND a scheduler.
    # Consider for yourself what the impact of this is,
    # and if you might want to change this (see lesson 4)
    search = HyperOptSearch()
    scheduler = AsyncHyperBandScheduler(
        time_attr="training_iteration",
        grace_period=1,
        reduction_factor=3,
        max_t=MAX_EPOCHS,
    )

    config = {
        "input_size": 3,
        "output_size": 20,
        "hidden_size": tune.randint(16, 128),
        "dropout": tune.uniform(0.0, 0.3),
        "num_layers": tune.randint(2, 5),
    }

    reporter = CLIReporter()
    reporter.add_metric_column("Accuracy")

    analysis = tune.run(
        train,
        config=config,
        metric="test_loss",
        mode="min",
        progress_reporter=reporter,
        storage_path=str(tune_dir),
        num_samples=NUM_SAMPLES,
        search_alg=search,
        scheduler=scheduler,
        verbose=1,
    )

    ray.shutdown()
