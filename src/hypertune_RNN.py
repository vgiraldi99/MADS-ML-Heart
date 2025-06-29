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

NUM_SAMPLES = 60
MAX_EPOCHS = 15
DATA_SET = 'arrhythmia'
CONCURRENT_TRIALS = 8
CLASS_WEIGHTS = torch.tensor([ #Using class weights to counter inbalance in the data. Using the inverse frequency method (1/%)
  1/0.8277, # class 0 is 82.77% of dataset
  1/0.0254, # class 1 is 2.539% of dataset
  1/0.0661, # class 2 is 6.611% of dataset
  1/0.0073, # class 3 is 0.732% of dataset
  1/0.0735, # class 4 is 7.345% of dataset
])


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

    traindataset = datasets.HeartDataset1D(trainfile, target="target")
    testdataset = datasets.HeartDataset1D(testfile, target="target")
    

    with FileLock(data_dir / ".lock"):
        # we lock the datadir to avoid parallel instances trying to
        # access the datadir
        trainstreamer = BaseDatastreamer(traindataset, preprocessor=BasePreprocessor(), batchsize=32)
        teststreamer = BaseDatastreamer(testdataset, preprocessor=BasePreprocessor(), batchsize=32)

    # we set up the metric
    # and create the model with the config
    accuracy = metrics.Accuracy()
    recall = metrics.Recall('macro')

    modeltype = settings.pop('model_type')
    if modeltype.lower() == 'gru':
      modelconfig = models.GRUSettings(**settings)
      model = models.GRUmodel(modelconfig)
    elif modeltype.lower() == 'lstm':
      modelconfig = models.LSTMSettings(**settings)
      model = models.LSTMmodel(modelconfig)
    
    


    trainersettings = TrainerSettings(
        epochs=MAX_EPOCHS,
        metrics=[accuracy, recall],
        logdir="logs/heart1D",
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
        loss_fn=torch.nn.CrossEntropyLoss(weight = CLASS_WEIGHTS),
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
        "input_size" : 1,
        "model_type": tune.choice(['gru', 'lstm']),
        "hidden_size" : tune.choice(valid_hidden_sizes), 
        "num_layers" : tune.randint(1, 5), #  Amount of convolutional layers to add
        "output_size" : 5, #  Amount of end classes to be determined 
        "attention" : True,
        "dropout" : tune.uniform(0.1, 0.5),
        "attention_dropout" : tune.uniform(0.1, 0.5),
    }

    reporter = CLIReporter()
    reporter.add_metric_column("Accuracy")
    reporter.add_metric_column("Recall_macro", "Recall")

    analysis = tune.run(
        train,
        config=config,
        metric="Recall_macro", #Changed to Recall
        mode="max", # Train the maximum recall
        progress_reporter=reporter,
        storage_path=str(tune_dir),
        num_samples=NUM_SAMPLES,
        search_alg=search,
        scheduler=scheduler,
        max_concurrent_trials=CONCURRENT_TRIALS,
        resources_per_trial={"cpu": 2},
        verbose=1,
    )

    ray.shutdown()

"""
Current best trial: 61060034 with test_loss=0.15648059157861605 and params={'input_size': 1, 'hidden_size': 192, 'num_layers': 4, 'output_size': 5, 'attention': True, 'dropout': 0.42292638494240864, 'attention_dropout': 0.4832250893451444}
╭──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ Trial name       status       model_type       hidden_size     num_layers     dropout     attention_dropout     iter     total time (s)     iterations     train_loss     test_loss     Accuracy │
├──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ train_4d94a60d   TERMINATED   gru                      192              3    0.353361              0.41629        15           3672.82              14       0.251198      0.23182      0.926389 │
│ train_c3c58177   TERMINATED   lstm                     264              4    0.192446              0.326913        3            929.168              2       0.66126       0.561791     0.846528 │
│ train_fe753e84   TERMINATED   gru                       48              1    0.231305              0.443152       15            863.964             14       0.307632      0.345045     0.915972 │
│ train_3620dbe3   TERMINATED   gru                      152              1    0.156574              0.360925        3            268.775              2       0.577201      0.578768     0.83125  │
│ train_5777f24e   TERMINATED   lstm                     216              1    0.229108              0.263595        3            267.593              2       0.6117        0.618398     0.822917 │
│ train_7d41aae6   TERMINATED   lstm                      64              2    0.303034              0.412668       15            789.422             14       0.420612      0.331241     0.907639 │
│ train_3aa36708   TERMINATED   gru                       72              2    0.412449              0.104202        3            255.091              2       0.559812      0.580241     0.834722 │
│ train_7a329d39   TERMINATED   gru                       80              2    0.117779              0.175434       15           1370.35              14       0.218476      0.23946      0.935417 │
│ train_cc1d6d99   TERMINATED   lstm                     184              3    0.215093              0.204642        3            468.079              2       0.669624      0.65427      0.832639 │
│ train_e2bc3091   TERMINATED   lstm                      64              3    0.24024               0.40524         3            170.567              2       0.586575      0.565796     0.829167 │
│ train_33cd39c6   TERMINATED   gru                      296              3    0.139459              0.374616        9           3757.23               8       0.288783      0.331095     0.915972 │
│ train_a30b07f3   TERMINATED   lstm                     224              2    0.317308              0.167875        9           1283.76               8       0.641746      0.683665     0.815278 │
│ train_69fedcdf   TERMINATED   lstm                     224              4    0.107514              0.288843        3            740.299              2       0.595726      0.644249     0.836111 │
│ train_3caad4a6   TERMINATED   lstm                     176              1    0.189063              0.136202        3            222.048              2       0.577423      0.58181      0.839583 │
│ train_64daf252   TERMINATED   lstm                     192              4    0.202386              0.225531        3            598.506              2       0.621534      0.616786     0.831944 │
│ train_7907905e   TERMINATED   lstm                      88              2    0.150161              0.371113        9            588.458              8       0.479247      0.51295      0.859722 │
│ train_1a0c0e38   TERMINATED   lstm                     152              2    0.108041              0.187812        3            289.365              2       0.607762      0.591588     0.823611 │
│ train_dd540fb9   TERMINATED   gru                       24              1    0.27116               0.422309        3            173.92               2       0.619745      0.595848     0.808333 │
│ train_17051d05   TERMINATED   gru                      280              4    0.304368              0.45263         3           1553.61               2       0.664768      0.663773     0.83125  │
│ train_3e8005fc   TERMINATED   gru                       64              2    0.331999              0.34219         3            254.379              2       0.564386      0.572218     0.816667 │
│ train_446f3c14   TERMINATED   gru                      208              2    0.491276              0.483186       15           2851.32              14       0.21066       0.185826     0.955556 │
│ train_24649a7f   TERMINATED   gru                       80              2    0.392039              0.494718       15           1469.18              14       0.226272      0.229239     0.941667 │
│ train_02490de9   TERMINATED   lstm                     240              2    0.461522              0.243306        9           1393.38               8       0.412827      0.3395       0.905556 │
│ train_2adb3428   TERMINATED   gru                      256              3    0.271305              0.103481        3           1029.93               2       0.536317      0.676802     0.821528 │
│ train_737c9aaa   TERMINATED   gru                      104              1    0.370224              0.298634       15           1054.21              14       0.274142      0.258659     0.926389 │
│ train_2a7cf49f   TERMINATED   lstm                      16              2    0.432732              0.146679        3            106.385              2       0.637488      0.643195     0.820139 │
│ train_6340526b   TERMINATED   gru                      104              1    0.368924              0.286717        3            207.728              2       0.605008      0.550819     0.830556 │
│ train_40bc429c   TERMINATED   gru                      120              1    0.495546              0.315387        3            222.955              2       0.591972      0.72154      0.822917 │
│ train_0feb7a98   TERMINATED   gru                      168              1    0.350878              0.257945        3            266.543              2       0.548596      0.653225     0.80625  │
│ train_38482a0b   TERMINATED   gru                      160              3    0.446916              0.214132        9           1746.24               8       0.244161      0.379934     0.904861 │
│ train_5a00da90   TERMINATED   gru                       80              2    0.386663              0.136526        9            741.025              8       0.285983      0.333035     0.913889 │
│ train_28be09f7   TERMINATED   gru                       80              3    0.398039              0.499898        9           1105.44               8       0.303911      0.308626     0.922917 │
│ train_5f96779f   TERMINATED   gru                      272              2    0.467341              0.172273       15           3715.02              14       0.218715      0.173226     0.956944 │
│ train_003e0048   TERMINATED   gru                       80              2    0.259124              0.457741       15           1389.05              14       0.270167      0.246611     0.935417 │
│ train_61060034   TERMINATED   gru                      192              4    0.422926              0.483225       15           3963.31              14       0.190083      0.156481     0.959028 │
│ train_b356b9b0   TERMINATED   gru                      248              3    0.340182              0.428425       15           3997.93              14       0.163194      0.160605     0.959028 │
│ train_cd3501fd   TERMINATED   gru                       40              3    0.29027               0.399183        3            307.614              2       0.589773      0.575406     0.819444 │
│ train_bb2051a5   TERMINATED   gru                      232              4    0.402074              0.478124        3           1179.95               2       0.601483      0.609894     0.811806 │
│ train_18a29602   TERMINATED   gru                      136              1    0.486736              0.495625       15           1247.33              14       0.2646        0.256293     0.928472 │
│ train_1b614f5c   TERMINATED   gru                      208              2    0.475714              0.471521       15           2750.96              14       0.225952      0.193806     0.95625  │
│ train_e8c5b6cc   TERMINATED   gru                      128              2    0.499254              0.387664       15           1788.56              14       0.227908      0.247489     0.938889 │
│ train_0787a593   TERMINATED   gru                       56              1    0.444763              0.350993        3            177.696              2       0.604349      0.596962     0.814583 │
│ train_e0e77d95   TERMINATED   gru                       96              2    0.372229              0.436883       15           1526.94              14       0.28278       0.23134      0.941667 │
│ train_d7c9c459   TERMINATED   gru                      112              1    0.431417              0.451615        9            662.97               8       0.346521      0.314505     0.909722 │
│ train_548f0f0d   TERMINATED   gru                        8              3    0.411774              0.493856        3            255.933              2       0.674015      0.664358     0.827083 │
│ train_f31e0983   TERMINATED   gru                      144              2    0.457234              0.321159       15           1581.59              14       0.202901      0.188068     0.952083 │
│ train_cbb79731   TERMINATED   gru                       32              1    0.325914              0.407998        3            181.107              2       0.621978      0.636085     0.800694 │
│ train_f5c99f77   TERMINATED   lstm                     288              2    0.476608              0.465525        3            627.094              2       0.620547      0.636816     0.794444 │
│ train_378f715c   TERMINATED   gru                      208              3    0.385159              0.37675         3            714.4                2       0.577214      0.504441     0.84375  │
│ train_a7e99fe2   TERMINATED   lstm                     264              2    0.359368              0.440219        3            531.875              2       0.550641      0.509904     0.852778 │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
"""