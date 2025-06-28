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

NUM_SAMPLES = 10
MAX_EPOCHS = 5
DATA_SET = 'arrhythmia'
CONCURRENT_TRIALS = 10


def train(settings: Dict):
    """
    The train function should receive a config file, which is a Dict
    ray will modify the values inside the config before it is passed to the train
    function.
    """
    from src import datasets, metrics
    
    # Use absolute paths so Ray workers can find the files
    script_dir = Path(__file__).parent  # Directory where this script is located
    data_dir = script_dir.parent / 'data'  # Go up one level to find data folder
    configfile = script_dir / "config.toml"  # Config file in same dir as script

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
    
    modelconfig = models.CNNSettings(**settings)
    model = models.ConvBlocks(modelconfig)


    trainersettings = TrainerSettings(
        epochs=MAX_EPOCHS,
        metrics=[accuracy, recall],
        logdir="logs/heart2D",
        train_steps=len(trainstreamer) // 15,  # type: ignore
        valid_steps=len(teststreamer) // 15,  # type: ignore
        reporttypes=[ReportTypes.RAY, ReportTypes.TENSORBOARD, ReportTypes.MLFLOW],
        scheduler_kwargs={"factor": 0.1, "patience": 5},
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
        time_attr="training_iteration",  # Stop based on epochs
        grace_period=3,      # Let each trial run for at least 3 epochs before considering stopping
        reduction_factor=3,  # Keep top 1/3 of trials, stop the rest
        max_t=MAX_EPOCHS,   # Maximum epochs any trial can run
    )
    valid_hidden_sizes = [i for i in range(8, 300) if i % 8 == 0]
    config = {
        "matrix_shape" : (16, 12), #  Shape of the insert matrix
        "in_channels" : 1,
        "hidden_size" : tune.choice(valid_hidden_sizes), 
        "num_layers" : tune.randint(1, 5), #  Amount of convolutional layers to add
        "num_classes" : 5, #  Amount of end classes to be determined 
        "attention" : True,
        "dense_activation" : tune.choice(['gelu', 'relu', 'leaky_relu'])
    }

    reporter = CLIReporter()
    reporter.add_metric_column("Accuracy")
    reporter.add_metric_column("Recall_macro", "Recall")

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
        max_concurrent_trials=CONCURRENT_TRIALS,
        resources_per_trial={"cpu": 2},
        verbose=0,
    )

    ray.shutdown()

"""

"""