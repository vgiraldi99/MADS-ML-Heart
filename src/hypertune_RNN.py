# Hypertune RNN
from pathlib import Path
from typing import Dict
import tomllib

import os
os.environ.pop('AIR_VERBOSITY', None)  # Remove AIR_VERBOSITY if it exists
os.environ['RAY_AIR_NEW_OUTPUT'] = '0' # According to issue #38202 (https://github.com/ray-project/ray/issues/38202) the AIR Output prevents custom logging (adding Recall for this reporting)

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

NUM_SAMPLES = 150
MAX_EPOCHS = 30
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

    modeltype = settings["model_type"]
    filtered_settings = {k: v for k, v in settings.items() if k != "model_type"}
    if modeltype.lower() == 'gru':
      modelconfig = models.GRUSettings(**filtered_settings)
      model = models.GRUmodel(modelconfig)
    elif modeltype.lower() == 'lstm':
      modelconfig = models.LSTMSettings(**filtered_settings)
      model = models.LSTMmodel(modelconfig)
    
    


    trainersettings = TrainerSettings(
        epochs=MAX_EPOCHS,
        metrics=[accuracy, recall],
        logdir="logs/heart1D",
        train_steps=len(trainstreamer) // 5,  # type: ignore
        valid_steps=len(teststreamer) // 5,  # type: ignore
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
        grace_period=5,      # Let each trial run for at least 3 epochs before considering stopping
        reduction_factor=3,  # Keep top 1/3 of trials, stop the rest
        max_t=MAX_EPOCHS,   # Maximum epochs any trial can run
    )
    valid_hidden_sizes = [i for i in range(8, 200) if i % 8 == 0]
    config = {
        "input_size" : 1,
        "output_size" : 5, #  Amount of end classes to be determined 
		"attention" : True,
		"model_type": tune.choice(['gru', 'lstm']),
        "hidden_size" : tune.choice(valid_hidden_sizes), 
        "num_layers" : tune.randint(1, 3), #  Amount of convolutional layers to add
        "dropout" : tune.uniform(0.1, 0.5),
        "attention_dropout" : tune.uniform(0.1, 0.5),
    }

    reporter = CLIReporter(metric_columns={"training_iteration": "iter",  # Correct metric name from Ray
                                           "time_total_s": "time (s)",    # Correct time metric name
                                           "Accuracy": "Accuracy", 
                                           "Recall_macro": "Recall", 
                                           "train_loss": "train loss", 
                                           "test_loss": "test loss"},
                          parameter_columns=["model_type", "hidden_size", "num_layers", "dropout", "attention_dropout"])
										   

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

== Status ==
Current time: 2025-06-29 20:45:25 (running for 09:31:34.34)
Using AsyncHyperBand: num_stopped=150
Bracket: Iter 15.000: 0.8705810160000363 | Iter 5.000: 0.7220816661357005
Logical resource usage: 2.0/24 CPUs, 0/0 GPUs
Current best trial: 3bbb4c55 with Recall_macro=0.9383543863247001 and parameters={'model_type': 'gru', 'hidden_size': 168, 'num_layers': 2, 'dropout': 0.4253390928530333, 'attention_dropout': 0.35820111410985095}
Result logdir: /tmp/ray/session_2025-06-29_11-13-49_864851_21033/artifacts/2025-06-29_11-13-51/train_2025-06-29_11-13-51/driver_artifacts
Number of trials: 150/150 (150 TERMINATED)
+----------------+------------+------------------+--------------+---------------+--------------+-----------+---------------------+--------+------------+------------+----------+--------------+-------------+
| Trial name     | status     | loc              | model_type   |   hidden_size |   num_layers |   dropout |   attention_dropout |   iter |   time (s) |   Accuracy |   Recall |   train loss |   test loss |
|----------------+------------+------------------+--------------+---------------+--------------+-----------+---------------------+--------+------------+------------+----------+--------------+-------------|
| train_e6fc75ad | TERMINATED | 172.17.0.2:22715 | lstm         |           120 |            2 |  0.401644 |            0.108317 |      5 |   1458.47  |   0.803079 | 0.368228 |     1.51787  |    1.51417  |
| train_b3b1bd13 | TERMINATED | 172.17.0.2:22795 | lstm         |            88 |            2 |  0.489598 |            0.421698 |      5 |    513.781 |   0.494256 | 0.489005 |     1.24969  |    1.18608  |
| train_81965a61 | TERMINATED | 172.17.0.2:22911 | lstm         |           184 |            2 |  0.19973  |            0.42613  |      5 |    903.566 |   0.422794 | 0.400715 |     1.36117  |    1.29416  |
| train_af42414c | TERMINATED | 172.17.0.2:22999 | lstm         |           152 |            1 |  0.152421 |            0.485227 |      5 |    533.531 |   0.726103 | 0.415462 |     1.44482  |    1.41221  |
| train_8d8db732 | TERMINATED | 172.17.0.2:23093 | gru          |            72 |            2 |  0.271068 |            0.258277 |     30 |   3708.4   |   0.837086 | 0.871951 |     0.31964  |    0.393473 |
| train_6df763d2 | TERMINATED | 172.17.0.2:23181 | gru          |            72 |            1 |  0.223715 |            0.202209 |     30 |   2484.59  |   0.88534  | 0.851412 |     0.372025 |    0.432057 |
| train_88cfb8d5 | TERMINATED | 172.17.0.2:23269 | gru          |           112 |            1 |  0.485221 |            0.387245 |      5 |    606.739 |   0.830423 | 0.438748 |     1.51749  |    1.4726   |
| train_9211938f | TERMINATED | 172.17.0.2:23358 | gru          |           128 |            2 |  0.437578 |            0.322517 |     30 |   5172.51  |   0.917509 | 0.919711 |     0.29647  |    0.279726 |
| train_703294c1 | TERMINATED | 172.17.0.2:23549 | gru          |           136 |            2 |  0.370162 |            0.357124 |     30 |   5356.87  |   0.904182 | 0.912899 |     0.245469 |    0.252495 |
| train_fcb6d705 | TERMINATED | 172.17.0.2:23628 | gru          |            88 |            2 |  0.369222 |            0.151461 |     30 |   3963.6   |   0.855928 | 0.915093 |     0.284134 |    0.284595 |
| train_6522d00f | TERMINATED | 172.17.0.2:23712 | lstm         |            16 |            1 |  0.119448 |            0.209352 |      5 |    229.302 |   0.761029 | 0.437063 |     1.41417  |    1.34417  |
| train_865ff470 | TERMINATED | 172.17.0.2:23826 | lstm         |            72 |            1 |  0.397449 |            0.231244 |      5 |    326.399 |   0.426241 | 0.445685 |     1.33995  |    1.28603  |
| train_d503118a | TERMINATED | 172.17.0.2:23907 | lstm         |           168 |            2 |  0.329206 |            0.364625 |      5 |    827.669 |   0.344669 | 0.634282 |     1.09801  |    0.909785 |
| train_b4622cc4 | TERMINATED | 172.17.0.2:24021 | gru          |           144 |            1 |  0.200464 |            0.487048 |     15 |   1864.99  |   0.727711 | 0.793187 |     0.524175 |    0.587518 |
| train_bbcfa57c | TERMINATED | 172.17.0.2:24129 | lstm         |           176 |            1 |  0.44488  |            0.433894 |      5 |    608.875 |   0.586627 | 0.444126 |     1.33646  |    1.2166   |
| train_7f506d0f | TERMINATED | 172.17.0.2:24238 | lstm         |            96 |            1 |  0.231647 |            0.427636 |      5 |    408.026 |   0.493336 | 0.44673  |     1.2862   |    1.22671  |
| train_a44fe3aa | TERMINATED | 172.17.0.2:24352 | lstm         |           144 |            2 |  0.499485 |            0.225642 |     15 |   2089.08  |   0.689108 | 0.810374 |     0.522411 |    0.511128 |
| train_95079fe7 | TERMINATED | 172.17.0.2:24436 | lstm         |           184 |            2 |  0.233255 |            0.158048 |      5 |    881.651 |   0.347426 | 0.622519 |     0.998984 |    0.901596 |
| train_70b5fbef | TERMINATED | 172.17.0.2:24556 | gru          |           176 |            2 |  0.134203 |            0.412759 |     30 |   6350.46  |   0.900965 | 0.909701 |     0.225593 |    0.321667 |
| train_a2110169 | TERMINATED | 172.17.0.2:24691 | lstm         |           120 |            2 |  0.373687 |            0.493512 |      5 |    633.255 |   0.839384 | 0.300123 |     1.58592  |    1.563    |
| train_87a3fbff | TERMINATED | 172.17.0.2:24769 | gru          |           104 |            1 |  0.180333 |            0.293112 |      5 |    505.92  |   0.689338 | 0.638892 |     1.00459  |    1.05948  |
| train_4f67815c | TERMINATED | 172.17.0.2:24906 | gru          |            40 |            1 |  0.303835 |            0.16811  |      5 |    360.431 |   0.540211 | 0.530636 |     1.13317  |    1.135    |
| train_2e36e000 | TERMINATED | 172.17.0.2:24997 | gru          |           160 |            1 |  0.256157 |            0.116674 |      5 |    606.374 |   0.431296 | 0.619659 |     0.971657 |    0.979863 |
| train_181c2d41 | TERMINATED | 172.17.0.2:25077 | gru          |            72 |            1 |  0.272656 |            0.268297 |     15 |   1243.69  |   0.754136 | 0.801375 |     0.55343  |    0.504066 |
| train_23b4f3c8 | TERMINATED | 172.17.0.2:25185 | gru          |            72 |            2 |  0.322222 |            0.27008  |     15 |   1834.01  |   0.865119 | 0.855649 |     0.413635 |    0.46556  |
| train_2ba7c29d | TERMINATED | 172.17.0.2:25289 | gru          |            24 |            1 |  0.27251  |            0.186496 |      5 |    332.074 |   0.578814 | 0.513958 |     1.16715  |    1.14677  |
| train_ea0082cb | TERMINATED | 172.17.0.2:25382 | gru          |            80 |            2 |  0.222426 |            0.325533 |      5 |    625.347 |   0.534007 | 0.65902  |     0.801652 |    0.830771 |
| train_b768e627 | TERMINATED | 172.17.0.2:25484 | gru          |             8 |            1 |  0.163222 |            0.253048 |      5 |    319.338 |   0.748621 | 0.508234 |     1.29201  |    1.243    |
| train_9220db39 | TERMINATED | 172.17.0.2:25561 | gru          |            56 |            2 |  0.355236 |            0.108885 |     15 |   1553.47  |   0.863051 | 0.844753 |     0.493839 |    0.410639 |
| train_e7eca458 | TERMINATED | 172.17.0.2:25678 | gru          |            88 |            2 |  0.428144 |            0.139703 |     15 |   1928.41  |   0.689338 | 0.824718 |     0.518621 |    0.520512 |
| train_f81b18a6 | TERMINATED | 172.17.0.2:25763 | gru          |            64 |            2 |  0.328363 |            0.137072 |     15 |   1630.8   |   0.870404 | 0.844586 |     0.451431 |    0.510239 |
| train_557fb8d5 | TERMINATED | 172.17.0.2:25849 | gru          |            48 |            2 |  0.411054 |            0.181595 |      5 |    507.479 |   0.546186 | 0.631532 |     0.915785 |    0.914735 |
| train_9bf95499 | TERMINATED | 172.17.0.2:25958 | gru          |           128 |            2 |  0.473064 |            0.320046 |     30 |   5032.72  |   0.921415 | 0.908138 |     0.232658 |    0.277138 |
| train_430ad25a | TERMINATED | 172.17.0.2:26111 | gru          |            88 |            2 |  0.461072 |            0.455589 |     30 |   4101.46  |   0.934743 | 0.90938  |     0.236539 |    0.299341 |
| train_6e49f370 | TERMINATED | 172.17.0.2:26222 | gru          |            32 |            2 |  0.389693 |            0.298064 |      5 |    483.105 |   0.586167 | 0.637097 |     1.11073  |    0.944228 |
| train_41cbfe5b | TERMINATED | 172.17.0.2:26308 | gru          |           128 |            2 |  0.433254 |            0.34569  |     30 |   4965.63  |   0.911994 | 0.897574 |     0.278606 |    0.367689 |
| train_d6d0f772 | TERMINATED | 172.17.0.2:26413 | gru          |           192 |            2 |  0.412455 |            0.321877 |      5 |   1164.08  |   0.526654 | 0.68384  |     0.857081 |    0.827301 |
| train_79951903 | TERMINATED | 172.17.0.2:26508 | gru          |           128 |            2 |  0.352801 |            0.391188 |     15 |   2508.78  |   0.865119 | 0.852431 |     0.52005  |    0.423014 |
| train_497e6270 | TERMINATED | 172.17.0.2:26623 | gru          |           152 |            2 |  0.466082 |            0.242791 |     30 |   5620.24  |   0.925322 | 0.931444 |     0.217921 |    0.252643 |
| train_d11b53ec | TERMINATED | 172.17.0.2:26724 | gru          |           112 |            2 |  0.301231 |            0.279029 |     30 |   4429.75  |   0.907399 | 0.921829 |     0.290928 |    0.220511 |
| train_a8fa10b7 | TERMINATED | 172.17.0.2:26853 | gru          |           168 |            2 |  0.382228 |            0.375362 |      5 |   1554.95  |   0.78079  | 0.424619 |     1.50512  |    1.48366  |
| train_aae29fde | TERMINATED | 172.17.0.2:27087 | gru          |            88 |            2 |  0.353235 |            0.215404 |     15 |   1983.47  |   0.912684 | 0.859477 |     0.46314  |    0.473573 |
| train_1b3ff01c | TERMINATED | 172.17.0.2:27165 | lstm         |            64 |            2 |  0.494404 |            0.460174 |      5 |    406.792 |   0.827436 | 0.293995 |     1.58125  |    1.58053  |
| train_c5cc9410 | TERMINATED | 172.17.0.2:27246 | gru          |           136 |            2 |  0.451623 |            0.331897 |     15 |   2652.66  |   0.828125 | 0.860838 |     0.388452 |    0.377079 |
| train_2d2421f3 | TERMINATED | 172.17.0.2:27372 | lstm         |            48 |            2 |  0.420537 |            0.400777 |      5 |    358.922 |   0.80216  | 0.351382 |     1.51545  |    1.53316  |
| train_f4ed5f3c | TERMINATED | 172.17.0.2:27489 | gru          |            16 |            2 |  0.396895 |            0.350725 |      5 |    453.184 |   0.356847 | 0.415032 |     1.29016  |    1.29786  |
| train_9b4f3c20 | TERMINATED | 172.17.0.2:27567 | lstm         |            96 |            2 |  0.480661 |            0.191267 |      5 |    465.469 |   0.423254 | 0.626846 |     1.09898  |    0.908476 |
| train_c849adfc | TERMINATED | 172.17.0.2:27697 | gru          |           184 |            2 |  0.101589 |            0.242009 |     15 |   3285.99  |   0.816636 | 0.861728 |     0.433033 |    0.467725 |
| train_47595727 | TERMINATED | 172.17.0.2:27776 | lstm         |           120 |            2 |  0.445578 |            0.373248 |      5 |    615.439 |   0.54159  | 0.571837 |     1.23798  |    1.03472  |
| train_c063a0cb | TERMINATED | 172.17.0.2:27874 | gru          |           104 |            1 |  0.366192 |            0.287407 |      5 |    494.759 |   0.595358 | 0.706605 |     0.868691 |    0.77376  |
| train_20b37526 | TERMINATED | 172.17.0.2:28004 | gru          |           160 |            2 |  0.286897 |            0.310193 |     30 |   5801.81  |   0.923483 | 0.924032 |     0.225524 |    0.242434 |
| train_3fa8e7fd | TERMINATED | 172.17.0.2:28086 | lstm         |           128 |            1 |  0.333986 |            0.129494 |      5 |    439.947 |   0.46898  | 0.603409 |     1.14587  |    0.962205 |
| train_9aeb98ab | TERMINATED | 172.17.0.2:28165 | gru          |            40 |            2 |  0.499078 |            0.151642 |      5 |    496.647 |   0.641085 | 0.682258 |     1.03075  |    0.878175 |
| train_45e092fb | TERMINATED | 172.17.0.2:28256 | gru          |            24 |            2 |  0.257337 |            0.450303 |      5 |    497.383 |   0.592601 | 0.5208   |     1.29857  |    1.23926  |
| train_48648c34 | TERMINATED | 172.17.0.2:28380 | lstm         |            80 |            1 |  0.340175 |            0.207944 |      5 |    336.061 |   0.599954 | 0.410365 |     1.36455  |    1.50083  |
| train_6e3c862c | TERMINATED | 172.17.0.2:28457 | gru          |           112 |            2 |  0.300573 |            0.279117 |     15 |   2255.34  |   0.831572 | 0.870308 |     0.391831 |    0.402628 |
| train_55a94331 | TERMINATED | 172.17.0.2:28546 | gru          |           112 |            2 |  0.311951 |            0.225739 |      5 |    749.416 |   0.673713 | 0.682272 |     0.957364 |    0.825402 |
| train_56bb004e | TERMINATED | 172.17.0.2:28635 | gru          |           112 |            1 |  0.28855  |            0.341079 |      5 |    512.857 |   0.569164 | 0.696872 |     0.946647 |    0.853422 |
| train_7bac7181 | TERMINATED | 172.17.0.2:28737 | gru          |           144 |            2 |  0.197085 |            0.30944  |     15 |   2709.97  |   0.816406 | 0.856412 |     0.387627 |    0.4884   |
| train_634e52df | TERMINATED | 172.17.0.2:28812 | lstm         |             8 |            2 |  0.155586 |            0.260482 |      5 |    250.158 |   0.362592 | 0.418219 |     1.28128  |    1.2697   |
| train_154c1d62 | TERMINATED | 172.17.0.2:28925 | gru          |           176 |            1 |  0.131938 |            0.365827 |     15 |   2042.8   |   0.910846 | 0.865659 |     0.457062 |    0.455126 |
| train_ef4bac52 | TERMINATED | 172.17.0.2:29010 | gru          |            32 |            2 |  0.183504 |            0.417008 |      5 |    501.192 |   0.771369 | 0.63829  |     1.04578  |    0.962964 |
| train_e00cc6ac | TERMINATED | 172.17.0.2:29112 | gru          |            56 |            2 |  0.252092 |            0.472382 |      5 |    594.429 |   0.487822 | 0.667846 |     0.950936 |    0.814369 |
| train_289a8644 | TERMINATED | 172.17.0.2:29208 | gru          |           152 |            2 |  0.21042  |            0.244185 |     30 |   5574.09  |   0.922794 | 0.928958 |     0.199568 |    0.239079 |
| train_b1be7fa9 | TERMINATED | 172.17.0.2:29309 | gru          |           152 |            1 |  0.105631 |            0.173976 |      5 |    595.954 |   0.68704  | 0.656263 |     0.924729 |    0.894059 |
| train_5a6a986b | TERMINATED | 172.17.0.2:29414 | gru          |           152 |            2 |  0.467476 |            0.278486 |     15 |   2846.74  |   0.845588 | 0.867888 |     0.398314 |    0.412776 |
| train_025978ad | TERMINATED | 172.17.0.2:29530 | gru          |           152 |            2 |  0.453726 |            0.305178 |     30 |   5659.44  |   0.899816 | 0.894361 |     0.289384 |    0.334125 |
| train_5c41396e | TERMINATED | 172.17.0.2:29645 | gru          |           112 |            2 |  0.434856 |            0.249582 |      5 |    752.309 |   0.579963 | 0.701527 |     0.834856 |    0.832464 |
| train_ecb8990d | TERMINATED | 172.17.0.2:29735 | gru          |           192 |            2 |  0.485587 |            0.197526 |      5 |   1144.87  |   0.50046  | 0.691813 |     0.850683 |    0.725014 |
| train_23373a00 | TERMINATED | 172.17.0.2:29857 | gru          |           136 |            2 |  0.406656 |            0.230278 |     30 |   5176.95  |   0.931066 | 0.933764 |     0.191357 |    0.211986 |
| train_939dfccc | TERMINATED | 172.17.0.2:29964 | gru          |           128 |            2 |  0.373039 |            0.268609 |      5 |    833.528 |   0.558364 | 0.621111 |     1.12229  |    0.978907 |
| train_d984b9c1 | TERMINATED | 172.17.0.2:30055 | gru          |            16 |            2 |  0.474611 |            0.33959  |      5 |    448.651 |   0.486443 | 0.511442 |     1.2344   |    1.13565  |
| train_0f6369bd | TERMINATED | 172.17.0.2:30178 | gru          |            96 |            2 |  0.387737 |            0.286301 |     15 |   2053.15  |   0.846737 | 0.833006 |     0.458237 |    0.484764 |
| train_81cbdbce | TERMINATED | 172.17.0.2:30262 | gru          |           184 |            2 |  0.315868 |            0.405794 |      5 |   1130.45  |   0.47932  | 0.693308 |     0.77824  |    0.758387 |
| train_5c098340 | TERMINATED | 172.17.0.2:30371 | gru          |           120 |            2 |  0.240449 |            0.386841 |     15 |   2426.93  |   0.904642 | 0.869437 |     0.468126 |    0.430567 |
| train_3bbb4c55 | TERMINATED | 172.17.0.2:30507 | gru          |           168 |            2 |  0.425339 |            0.358201 |     30 |   6257.63  |   0.938649 | 0.938354 |     0.195229 |    0.216343 |
| train_5d41b787 | TERMINATED | 172.17.0.2:30622 | lstm         |            64 |            2 |  0.462016 |            0.236334 |      5 |    366.011 |   0.8233   | 0.288603 |     1.57664  |    1.57311  |
| train_640f3b90 | TERMINATED | 172.17.0.2:30741 | gru          |            40 |            2 |  0.438809 |            0.433774 |      5 |    534.36  |   0.566866 | 0.64464  |     0.939483 |    0.988885 |
| train_73a7017c | TERMINATED | 172.17.0.2:30851 | gru          |           160 |            2 |  0.292212 |            0.219469 |     15 |   2887.07  |   0.901195 | 0.811538 |     0.363844 |    0.654536 |
| train_f7d567e7 | TERMINATED | 172.17.0.2:30945 | gru          |           160 |            2 |  0.280843 |            0.315271 |     30 |   5883.94  |   0.942555 | 0.916772 |     0.217212 |    0.238655 |
| train_90ae738e | TERMINATED | 172.17.0.2:31032 | gru          |           104 |            2 |  0.342529 |            0.262281 |     15 |   2210.73  |   0.783318 | 0.8544   |     0.379533 |    0.411078 |
| train_f18086ef | TERMINATED | 172.17.0.2:31180 | lstm         |           160 |            2 |  0.219203 |            0.164263 |      5 |    997.009 |   0.715303 | 0.441017 |     1.49008  |    1.47674  |
| train_2184883d | TERMINATED | 172.17.0.2:31301 | gru          |           152 |            2 |  0.212127 |            0.246044 |      5 |    964.516 |   0.389706 | 0.647603 |     0.884173 |    0.833948 |
| train_9be9b10d | TERMINATED | 172.17.0.2:31441 | gru          |           152 |            2 |  0.180147 |            0.331449 |     30 |   5639.76  |   0.930377 | 0.904201 |     0.248969 |    0.332707 |
| train_4837a3bb | TERMINATED | 172.17.0.2:31551 | gru          |           152 |            2 |  0.260913 |            0.299262 |      5 |    952.926 |   0.560432 | 0.714112 |     0.740965 |    0.765618 |
| train_ede62125 | TERMINATED | 172.17.0.2:31636 | gru          |           160 |            1 |  0.243056 |            0.202118 |      5 |    616.224 |   0.482077 | 0.629863 |     0.995449 |    0.855412 |
| train_19ace1b9 | TERMINATED | 172.17.0.2:31743 | lstm         |            48 |            2 |  0.16645  |            0.18382  |      5 |    322.903 |   0.815028 | 0.356198 |     1.51636  |    1.52094  |
| train_2ad0f162 | TERMINATED | 172.17.0.2:31825 | gru          |            24 |            2 |  0.404058 |            0.233446 |      5 |    467.834 |   0.514706 | 0.573253 |     1.1618   |    1.09029  |
| train_a1627db3 | TERMINATED | 172.17.0.2:31937 | gru          |           136 |            2 |  0.133888 |            0.215431 |     15 |   2622.99  |   0.865579 | 0.859072 |     0.416051 |    0.426689 |
| train_619c1c62 | TERMINATED | 172.17.0.2:32021 | gru          |           136 |            1 |  0.112264 |            0.122362 |      5 |    544.557 |   0.565487 | 0.615655 |     1.10153  |    1.00312  |
| train_fa0943fa | TERMINATED | 172.17.0.2:32100 | lstm         |            72 |            2 |  0.362504 |            0.174538 |      5 |    405.89  |   0.294347 | 0.384576 |     1.48909  |    1.30499  |
| train_0000daf9 | TERMINATED | 172.17.0.2:32201 | gru          |           136 |            2 |  0.412666 |            0.149316 |     30 |   5166.06  |   0.921415 | 0.930404 |     0.212431 |    0.231665 |
| train_244d67a7 | TERMINATED | 172.17.0.2:32289 | gru          |           152 |            2 |  0.209023 |            0.254478 |      5 |    953.837 |   0.502757 | 0.637145 |     0.773223 |    0.894777 |
| train_d33e7fa8 | TERMINATED | 172.17.0.2:32389 | gru          |            80 |            2 |  0.145021 |            0.108615 |      5 |    605.999 |   0.440717 | 0.530431 |     1.16322  |    1.17868  |
| train_35889ef8 | TERMINATED | 172.17.0.2:32481 | gru          |           144 |            1 |  0.190753 |            0.193058 |     15 |   1738.78  |   0.725184 | 0.819511 |     0.520318 |    0.471198 |
| train_049a4f4c | TERMINATED | 172.17.0.2:32616 | lstm         |             8 |            2 |  0.492267 |            0.242997 |      5 |    248.284 |   0.723346 | 0.505028 |     1.3219   |    1.29386  |
| train_5df0d04b | TERMINATED | 172.17.0.2:32713 | gru          |            56 |            2 |  0.400617 |            0.224346 |     15 |   1655.71  |   0.749311 | 0.824617 |     0.51881  |    0.436857 |
| train_3c5bae05 | TERMINATED | 172.17.0.2:32798 | gru          |           176 |            2 |  0.381072 |            0.206611 |      5 |   1062.57  |   0.504136 | 0.71354  |     0.733808 |    0.705322 |
| train_7d0e2909 | TERMINATED | 172.17.0.2:32983 | gru          |           136 |            1 |  0.230146 |            0.158391 |      5 |    551.853 |   0.586857 | 0.69771  |     1.04335  |    0.932805 |
| train_50b489ea | TERMINATED | 172.17.0.2:33066 | gru          |           192 |            2 |  0.265427 |            0.292012 |     15 |   3384.84  |   0.860524 | 0.83925  |     0.488483 |    0.498375 |
| train_de98170c | TERMINATED | 172.17.0.2:33144 | lstm         |           168 |            2 |  0.423632 |            0.378366 |      5 |    837.739 |   0.825597 | 0.292034 |     1.5883   |    1.58579  |
| train_ee52f3bf | TERMINATED | 172.17.0.2:33241 | gru          |           168 |            2 |  0.453582 |            0.357484 |      5 |   1494.19  |   0.482307 | 0.562716 |     1.12561  |    1.03851  |
| train_a41a3fea | TERMINATED | 172.17.0.2:33362 | gru          |           168 |            2 |  0.418102 |            0.36253  |      5 |   1049.91  |   0.397059 | 0.56766  |     1.40488  |    1.11485  |
| train_59167d86 | TERMINATED | 172.17.0.2:33439 | gru          |            32 |            1 |  0.47941  |            0.328691 |      5 |    361.726 |   0.526654 | 0.560312 |     1.15772  |    1.16638  |
| train_606445f3 | TERMINATED | 172.17.0.2:33559 | gru          |            88 |            2 |  0.442997 |            0.352    |     15 |   1995.82  |   0.86489  | 0.858614 |     0.432747 |    0.404573 |
| train_37503590 | TERMINATED | 172.17.0.2:33637 | lstm         |           168 |            2 |  0.428995 |            0.398962 |      5 |   1885.87  |   0.282858 | 0.546827 |     1.03678  |    1.01006  |
| train_62379c04 | TERMINATED | 172.17.0.2:33706 | gru          |            16 |            2 |  0.407907 |            0.276103 |      5 |    427.36  |   0.421186 | 0.487402 |     1.24303  |    1.19295  |
| train_d812e089 | TERMINATED | 172.17.0.2:33848 | gru          |            96 |            2 |  0.396031 |            0.450254 |     15 |   2152.79  |   0.86489  | 0.851929 |     0.414106 |    0.452056 |
| train_fce2e5e1 | TERMINATED | 172.17.0.2:33952 | gru          |           184 |            2 |  0.464721 |            0.318676 |      5 |   1161.14  |   0.53171  | 0.514952 |     1.09038  |    1.15696  |
| train_be14f33d | TERMINATED | 172.17.0.2:34043 | gru          |           120 |            1 |  0.357824 |            0.422285 |      5 |    537.712 |   0.450827 | 0.52813  |     1.17743  |    1.10747  |
| train_8b730c07 | TERMINATED | 172.17.0.2:34182 | lstm         |            64 |            2 |  0.34544  |            0.258991 |      5 |    368.159 |   0.596507 | 0.612805 |     1.09876  |    1.05666  |
| train_e0f381a0 | TERMINATED | 172.17.0.2:34264 | gru          |           104 |            2 |  0.323854 |            0.342298 |     30 |   4377.49  |   0.928998 | 0.91946  |     0.220211 |    0.247195 |
| train_3b1b444f | TERMINATED | 172.17.0.2:34381 | gru          |           136 |            2 |  0.498286 |            0.13939  |     30 |   5047.22  |   0.872243 | 0.904901 |     0.279755 |    0.340304 |
| train_5e90c19a | TERMINATED | 172.17.0.2:34468 | gru          |           168 |            2 |  0.385674 |            0.478611 |      5 |   1064.58  |   0.710708 | 0.61279  |     1.00983  |    1.08808  |
| train_83659139 | TERMINATED | 172.17.0.2:34549 | gru          |            40 |            1 |  0.471677 |            0.101541 |      5 |    337.753 |   0.532858 | 0.53264  |     1.14713  |    1.14617  |
| train_430b74da | TERMINATED | 172.17.0.2:34643 | lstm         |            72 |            2 |  0.447628 |            0.285117 |      5 |    416.773 |   0.50046  | 0.438761 |     1.35583  |    1.28471  |
| train_f3c7a3de | TERMINATED | 172.17.0.2:34739 | gru          |            48 |            2 |  0.378861 |            0.302859 |     15 |   1582.43  |   0.628217 | 0.738065 |     0.656223 |    0.663193 |
| train_dd706995 | TERMINATED | 172.17.0.2:34836 | gru          |            24 |            2 |  0.488627 |            0.230217 |      5 |    460.159 |   0.571002 | 0.479088 |     1.22015  |    1.15426  |
| train_a398e663 | TERMINATED | 172.17.0.2:34933 | gru          |            80 |            2 |  0.459089 |            0.269669 |      5 |    632.449 |   0.59352  | 0.688274 |     0.864064 |    0.840464 |
| train_9d6ad5da | TERMINATED | 172.17.0.2:35027 | gru          |           144 |            2 |  0.43265  |            0.385181 |     15 |   2726.41  |   0.802619 | 0.86469  |     0.403094 |    0.375994 |
| train_97a58037 | TERMINATED | 172.17.0.2:35110 | lstm         |             8 |            1 |  0.395159 |            0.336642 |      5 |    248.692 |   0.525506 | 0.50567  |     1.34242  |    1.32436  |
| train_cfdd8699 | TERMINATED | 172.17.0.2:35201 | gru          |           176 |            2 |  0.312822 |            0.369365 |     15 |   3186.27  |   0.859605 | 0.873952 |     0.408548 |    0.424054 |
| train_872a1edd | TERMINATED | 172.17.0.2:35297 | gru          |           136 |            2 |  0.336138 |            0.443631 |     15 |   2714.57  |   0.848116 | 0.86136  |     0.426334 |    0.416594 |
| train_58cd9750 | TERMINATED | 172.17.0.2:35379 | gru          |           192 |            2 |  0.369333 |            0.173518 |     15 |   3440.11  |   0.805377 | 0.849636 |     0.336004 |    0.475853 |
| train_3310a5ea | TERMINATED | 172.17.0.2:35472 | gru          |            56 |            2 |  0.482697 |            0.466117 |      5 |    589.652 |   0.560432 | 0.693545 |     1.00412  |    0.877934 |
| train_6322be37 | TERMINATED | 172.17.0.2:35622 | gru          |            32 |            1 |  0.349817 |            0.494859 |      5 |    392.828 |   0.622702 | 0.581912 |     1.16984  |    1.14649  |
| train_10d3ce6b | TERMINATED | 172.17.0.2:35701 | lstm         |            88 |            2 |  0.422598 |            0.238251 |      5 |    451.854 |   0.521829 | 0.426186 |     1.293    |    1.27695  |
| train_d2b00235 | TERMINATED | 172.17.0.2:35823 | gru          |            96 |            2 |  0.414705 |            0.312726 |     15 |   2127.09  |   0.904871 | 0.865706 |     0.411247 |    0.424651 |
| train_0c1568c3 | TERMINATED | 172.17.0.2:35908 | gru          |           128 |            2 |  0.437982 |            0.295009 |     15 |   2530.06  |   0.857537 | 0.854515 |     0.446482 |    0.439906 |
| train_d6260799 | TERMINATED | 172.17.0.2:36108 | gru          |           168 |            2 |  0.375829 |            0.215577 |      5 |   1024.3   |   0.662684 | 0.718647 |     0.861315 |    0.813894 |
| train_58caf75c | TERMINATED | 172.17.0.2:36219 | gru          |            16 |            2 |  0.362584 |            0.187331 |      5 |    428.169 |   0.595358 | 0.496621 |     1.19865  |    1.22417  |
| train_19641216 | TERMINATED | 172.17.0.2:36323 | lstm         |           184 |            1 |  0.455509 |            0.198056 |      5 |    673.214 |   0.783318 | 0.383768 |     1.52171  |    1.49926  |
| train_8226485e | TERMINATED | 172.17.0.2:36408 | gru          |           152 |            2 |  0.405747 |            0.412663 |      5 |    977.573 |   0.488281 | 0.7183   |     0.809068 |    0.662833 |
| train_c654bd13 | TERMINATED | 172.17.0.2:36500 | gru          |            64 |            2 |  0.473932 |            0.148316 |      5 |    572.792 |   0.730699 | 0.713899 |     0.8414   |    0.833303 |
| train_5fa5791b | TERMINATED | 172.17.0.2:36602 | gru          |           120 |            2 |  0.498575 |            0.252034 |     15 |   2359.12  |   0.888327 | 0.844022 |     0.422794 |    0.473243 |
| train_fb186e38 | TERMINATED | 172.17.0.2:36691 | gru          |           104 |            2 |  0.279273 |            0.122695 |     15 |   2128.61  |   0.763097 | 0.860266 |     0.402195 |    0.386818 |
| train_089e9773 | TERMINATED | 172.17.0.2:36778 | lstm         |            40 |            1 |  0.42465  |            0.324023 |      5 |    286.915 |   0.689798 | 0.697675 |     0.968272 |    0.896904 |
| train_3edbded0 | TERMINATED | 172.17.0.2:36876 | gru          |           136 |            2 |  0.444873 |            0.265522 |      5 |    871.074 |   0.490579 | 0.702691 |     0.826463 |    0.806753 |
| train_980e7f07 | TERMINATED | 172.17.0.2:36962 | gru          |           112 |            2 |  0.394137 |            0.398271 |     15 |   2264.85  |   0.878906 | 0.865099 |     0.418319 |    0.40158  |
| train_022b81bb | TERMINATED | 172.17.0.2:37045 | gru          |            48 |            2 |  0.467655 |            0.352706 |      5 |    546.365 |   0.467142 | 0.678081 |     0.940152 |    0.837275 |
| train_eda4d979 | TERMINATED | 172.17.0.2:37114 | gru          |            24 |            2 |  0.330891 |            0.429866 |      5 |    495.327 |   0.428079 | 0.472611 |     1.19558  |    1.26367  |
| train_ae271735 | TERMINATED | 172.17.0.2:37231 | lstm         |           168 |            1 |  0.307321 |            0.221291 |      5 |    514.976 |   0.327436 | 0.438157 |     1.28799  |    1.27536  |
| train_1241e83d | TERMINATED | 172.17.0.2:37316 | gru          |            80 |            2 |  0.320825 |            0.129918 |      5 |    603.046 |   0.427619 | 0.700692 |     0.682884 |    0.733828 |
| train_93d145f2 | TERMINATED | 172.17.0.2:37429 | gru          |           152 |            2 |  0.390188 |            0.208012 |      5 |    937.57  |   0.54182  | 0.70982  |     0.840479 |    0.789816 |
| train_32edb2d3 | TERMINATED | 172.17.0.2:37510 | gru          |            72 |            2 |  0.479127 |            0.275964 |     15 |   1749.03  |   0.837776 | 0.857164 |     0.426693 |    0.436306 |
| train_79f344ee | TERMINATED | 172.17.0.2:37597 | gru          |           136 |            2 |  0.412166 |            0.148643 |     15 |   2158.43  |   0.907858 | 0.869847 |     0.403792 |    0.4333   |
| train_394b1231 | TERMINATED | 172.17.0.2:37687 | gru          |           136 |            2 |  0.402022 |            0.178633 |     30 |   3694.44  |   0.927849 | 0.913501 |     0.298284 |    0.284039 |
| train_34170a7f | TERMINATED | 172.17.0.2:37776 | gru          |           136 |            2 |  0.448824 |            0.197493 |      5 |    886.476 |   0.782858 | 0.428737 |     1.48091  |    1.49239  |
| train_11e1711a | TERMINATED | 172.17.0.2:37928 | gru          |           136 |            2 |  0.433842 |            0.16358  |      5 |    779.861 |   0.572151 | 0.713449 |     0.710258 |    0.719    |
| train_a6089e30 | TERMINATED | 172.17.0.2:38036 | gru          |           144 |            2 |  0.41498  |            0.228206 |     30 |   3338.94  |   0.932445 | 0.935823 |     0.220269 |    0.206558 |
+----------------+------------+------------------+--------------+---------------+--------------+-----------+---------------------+--------+------------+------------+----------+--------------+-------------+

"""