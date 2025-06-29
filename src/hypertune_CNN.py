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
CONCURRENT_TRIALS = 10
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
    valid_hidden_sizes = [i for i in range(40, 150) if i % 8 == 0] #Trail 1: was 8 - 400
    config = {
        "matrix_shape" : (16, 12), #  Shape of the insert matrix
        "in_channels" : 1,
        "num_classes" : 5, #  Amount of end classes to be determined 
        "attention" : True,
        "hidden_size" : tune.choice(valid_hidden_sizes), 
        "num_layers" : tune.randint(1, 3), #  Amount of convolutional layers to add
        "dense_activation" : tune.choice(['gelu', 'relu', 'leaky_relu'])
    }

    reporter = CLIReporter(metric_columns={"training_iteration": "iter",  # Correct metric name from Ray
                                           "time_total_s": "time (s)",    # Correct time metric name
                                           "Accuracy": "Accuracy", 
                                           "Recall_macro": "Recall", 
                                           "train_loss": "train loss", 
                                           "test_loss": "test loss"},
                          parameter_columns=["hidden_size", "num_layers", "dense_activation"])

    analysis = tune.run(
        train,
        config=config,
        metric="Recall_macro", #Changed to recall
        mode="max", # Maximum recall OVER (0.)9000
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
Trial status: 150 TERMINATED
Current time: 2025-06-28 23:07:03. Total running time: 3hr 40min 4s
Logical resource usage: 2.0/24 CPUs, 0/0 GPUs
MAX_EPOCH = 25
NUM_EXP = 150
Current best trial: 1651a5b9 with Recall_macro=0.9473663567187807 and params={'matrix_shape': (16, 12), 'in_channels': 1, 'hidden_size': 80, 'num_layers': 1, 'num_classes': 5, 'attention': True, 'dense_activation': 'leaky_relu'}
╭──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ Trial name       status         hidden_size     num_layers   dense_activation       iter     total time (s)     iterations     train_loss     test_loss     Accuracy │
├──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ train_4726788f   TERMINATED             232              1   gelu                      9          1684.83                8       0.422617      0.390377    0.864583  │
│ train_823e3755   TERMINATED             128              2   gelu                      3           183.3                 2       1.02584       0.90342     0.644444  │
│ train_05a53aa2   TERMINATED             168              1   gelu                     25          2910.67               24       0.29884       0.252084    0.925     │
│ train_99bfe8f6   TERMINATED             104              2   leaky_relu               25          1494.87               24       0.274526      0.283891    0.902083  │
│ train_ac1438d6   TERMINATED             296              2   leaky_relu                3          1089.14                2       1.58524       1.55519     0.831944  │
│ train_d8b09098   TERMINATED              24              4   gelu                     25           383.463              24       0.526352      0.470657    0.85625   │
│ train_d99ae064   TERMINATED              72              4   gelu                      3           134.803               2       1.58051       1.59503     0.80625   │
│ train_b0e8e818   TERMINATED             216              2   relu                      3           594.164               2       1.47092       1.4482      0.611111  │
│ train_89819018   TERMINATED              40              2   gelu                     25           445.069              24       0.378614      0.230178    0.905556  │
│ train_cb9e6edd   TERMINATED             120              4   leaky_relu                3           259.939               2       1.58866       1.61541     0.833333  │
│ train_e9d5a266   TERMINATED               8              3   relu                      3            28.3187              2       1.55148       1.61991     0.0680556 │
│ train_882a3f89   TERMINATED             256              3   leaky_relu                3           841.811               2       1.57048       1.55296     0.830556  │
│ train_592628b9   TERMINATED               8              4   leaky_relu                3            32.0433              2       1.5907        1.58173     0.830556  │
│ train_00f68499   TERMINATED             296              3   leaky_relu                3          1212.67                2       1.57837       1.58318     0.822222  │
│ train_edcf0bb0   TERMINATED             136              2   leaky_relu                3           282.16                2       1.1551        1.12427     0.420833  │
│ train_be3f962b   TERMINATED              80              1   gelu                     25           821.95               24       0.213033      0.328728    0.920139  │
│ train_bd7667f0   TERMINATED             168              4   relu                      3           480.939               2       1.59117       1.58756     0.834028  │
│ train_5df25dfa   TERMINATED              96              2   relu                      3           142.93                2       1.04128       1.34002     0.263889  │
│ train_47c705fe   TERMINATED             272              3   relu                      3           999.135               2       1.55899       1.63788     0.805556  │
│ train_a68edac2   TERMINATED             264              3   relu                      3           989.944               2       1.58499       1.49555     0.772222  │
│ train_b0127df7   TERMINATED              40              1   gelu                     25           406.502              24       0.240346      0.447885    0.90625   │
│ train_2377cb52   TERMINATED              24              2   gelu                      3            41.4683              2       1.21067       0.983604    0.629167  │
│ train_3c3bd97f   TERMINATED              64              1   gelu                     25           573.052              24       0.226009      0.257135    0.90625   │
│ train_fb0aa164   TERMINATED              24              3   gelu                      3            49.8724              2       1.58591       1.56723     0.820833  │
│ train_1cece452   TERMINATED             240              2   gelu                      3           676.032               2       1.58033       1.59066     0.833333  │
│ train_2c07452e   TERMINATED              88              1   gelu                     25          1071.07               24       0.222101      0.179867    0.940278  │
│ train_5c20cb65   TERMINATED              80              1   gelu                      9           295.674               8       0.402063      0.42953     0.857639  │
│ train_6d38931e   TERMINATED             288              1   gelu                      9          2507.41                8       0.452414      0.349749    0.859722  │
│ train_20c6d6ff   TERMINATED             104              1   leaky_relu                9           502.266               8       0.380632      0.483503    0.825     │
│ train_8bdb6a64   TERMINATED             200              1   leaky_relu               25          4151.62               24       0.178776      0.191099    0.93125   │
│ train_d6b91317   TERMINATED              56              2   leaky_relu                3            84.8597              2       1.3649        1.23418     0.465278  │
│ train_d3a6414f   TERMINATED              64              1   leaky_relu                9           229.974               8       0.398763      0.445982    0.728472  │
│ train_21b40199   TERMINATED             224              2   leaky_relu                3           620.491               2       1.3548        1.28418     0.285417  │
│ train_de698581   TERMINATED             112              1   leaky_relu                9           507.881               8       0.416776      0.379644    0.864583  │
│ train_bd87f77a   TERMINATED             184              1   gelu                     25          3492.12               24       0.227183      0.265879    0.919444  │
│ train_a720e088   TERMINATED              32              2   leaky_relu                3            46.9571              2       1.23109       1.13345     0.564583  │
│ train_f4e99674   TERMINATED             208              2   gelu                      3           551.815               2       1.57993       1.49317     0.722917  │
│ train_ff3ab426   TERMINATED             192              1   gelu                      9          1165.68                8       0.315471      0.423216    0.870833  │
│ train_74ee030f   TERMINATED             248              2   relu                      3           760.824               2       1.47451       1.47607     0.521528  │
│ train_8f8bf8f1   TERMINATED             144              3   leaky_relu                3           314.846               2       1.55728       1.57834     0.825     │
│ train_f2da4fa5   TERMINATED              88              1   gelu                      9           378.283               8       0.4239        0.44364     0.835417  │
│ train_e10c88b1   TERMINATED              48              1   gelu                      9           150.936               8       0.433009      0.499926    0.825694  │
│ train_f6aa34a2   TERMINATED              16              1   gelu                      3            26.0471              2       0.745098      0.721113    0.815278  │
│ train_b0e54918   TERMINATED             176              1   gelu                      3           353.113               2       0.717187      0.582563    0.534028  │
│ train_068c4655   TERMINATED              64              1   gelu                      3            74.1035              2       0.732227      0.898451    0.6875    │
│ train_75113db5   TERMINATED             152              1   gelu                      3           295.681               2       0.759777      0.722359    0.672917  │
│ train_0531adc6   TERMINATED             280              2   gelu                      3           967.021               2       1.22921       1.11682     0.353472  │
│ train_78ca583c   TERMINATED             160              4   gelu                      3           398.035               2       1.58032       1.59167     0.832639  │
│ train_3346bc0e   TERMINATED              88              1   relu                      9           383.228               8       0.425811      0.525378    0.852778  │
│ train_59df60f1   TERMINATED             216              2   gelu                      3           611.765               2       1.59004       1.55616     0.0541667 │
│ train_123ba70f   TERMINATED              88              1   gelu                      3           133.168               2       0.629558      0.642328    0.665972  │
│ train_62a25b64   TERMINATED             232              3   relu                      3           780.673               2       1.58425       1.57661     0.838889  │
│ train_aa52ba79   TERMINATED              96              2   gelu                      3           142.886               2       1.01843       0.801437    0.681944  │
│ train_e966ee06   TERMINATED             128              1   gelu                     25          1723.07               24       0.170075      0.181159    0.934028  │
│ train_6d975f72   TERMINATED              72              4   relu                      3           145.797               2       1.58132       1.5917      0.824306  │
│ train_9c7d12bf   TERMINATED             256              3   gelu                      3           903.225               2       1.58034       1.56931     0.84375   │
│ train_005c4ac0   TERMINATED              64              2   gelu                      3            85.2968              2       1.20444       1.11508     0.529861  │
│ train_da2c802c   TERMINATED             136              1   gelu                      3           247.365               2       1.59096       1.60495     0.802083  │
│ train_d64c244f   TERMINATED             240              2   relu                      3           724.454               2       1.50612       1.42749     0.675694  │
│ train_3a070676   TERMINATED             296              1   gelu                     25          8138.24               24       0.25677       0.208805    0.911111  │
│ train_b5fb7712   TERMINATED             120              1   gelu                      9           648.126               8       0.351572      0.529933    0.885417  │
│ train_5dde3d50   TERMINATED               8              3   relu                      3            29.3437              2       1.51666       1.48596     0.615972  │
│ train_384b5890   TERMINATED             168              4   gelu                      3           547.35                2       1.5829        1.57245     0.820833  │
│ train_cb7da10c   TERMINATED             288              2   gelu                      3          1046.53                2       1.58341       1.62904     0.0729167 │
│ train_c184abe2   TERMINATED             264              1   gelu                     25          6601.91               24       0.174558      0.215005    0.947917  │
│ train_5e6ab341   TERMINATED             104              2   leaky_relu                3           215.688               2       1.1711        1.00746     0.436111  │
│ train_81313467   TERMINATED             200              1   leaky_relu                3           521.217               2       0.849904      0.795702    0.663889  │
│ train_f638de99   TERMINATED             104              2   leaky_relu                3           212.159               2       1.55918       1.62176     0.826389  │
│ train_ab0495d5   TERMINATED              40              1   leaky_relu               25           475.957              24       0.224444      0.199548    0.91875   │
│ train_a9821b5d   TERMINATED              56              3   leaky_relu                3           119.169               2       1.48456       1.44482     0.772917  │
│ train_7c9444ba   TERMINATED             272              1   leaky_relu                9          2444.55                8       0.40013       0.491719    0.836806  │
│ train_2bcf5dfd   TERMINATED             224              2   leaky_relu                3           683.362               2       1.28525       1.15742     0.396528  │
│ train_b2defb45   TERMINATED             112              1   leaky_relu               25          1511.07               24       0.22793       0.429559    0.915278  │
│ train_3fb68f89   TERMINATED              40              1   leaky_relu                9           178.781               8       0.386923      0.436644    0.831944  │
│ train_96ded705   TERMINATED              40              1   leaky_relu               25           492.268              24       0.235658      0.350774    0.858333  │
│ train_27bde57f   TERMINATED             128              1   leaky_relu                9           661.469               8       0.423247      0.337707    0.844444  │
│ train_76e29a7b   TERMINATED             184              1   gelu                      3           438.048               2       0.767495      0.72663     0.576389  │
│ train_b1e31716   TERMINATED             128              1   relu                      9           667.628               8       0.431351      0.575158    0.802778  │
│ train_5905b489   TERMINATED             128              1   gelu                      9           671.698               8       0.495964      0.563801    0.759028  │
│ train_005ef51e   TERMINATED             248              1   leaky_relu                9          2042.89                8       0.421885      0.497447    0.836806  │
│ train_9f730958   TERMINATED             192              1   gelu                      9          1237.86                8       0.414029      0.566949    0.775694  │
│ train_0a079443   TERMINATED             208              1   leaky_relu                9          1452.46                8       0.41275       0.413968    0.8625    │
│ train_2f1ec5b2   TERMINATED              32              1   gelu                      3            45.3525              2       0.638686      0.699099    0.675694  │
│ train_625128ec   TERMINATED             144              1   leaky_relu               25          2008.14               24       0.220016      0.25702     0.927778  │
│ train_64145198   TERMINATED             200              2   leaky_relu                3           559.301               2       1.09166       1.40115     0.390278  │
│ train_355e5aaa   TERMINATED              80              1   leaky_relu                3           104.137               2       0.817739      0.65972     0.679167  │
│ train_9dcafbfe   TERMINATED             200              1   leaky_relu                3           486.013               2       0.732897      0.666807    0.645833  │
│ train_05d82c3e   TERMINATED             176              2   leaky_relu                3           407.43                2       1.04841       0.956166    0.59375   │
│ train_56894dba   TERMINATED              48              1   leaky_relu                3            54.4232              2       0.681432      0.606273    0.68125   │
│ train_6e25a211   TERMINATED             152              1   leaky_relu                3           296.457               2       0.678167      0.607957    0.666667  │
│ train_1476b6b3   TERMINATED              16              2   leaky_relu                3            25.0227              2       1.08431       1.00515     0.579167  │
│ train_bf83cace   TERMINATED             160              4   relu                      3           393.345               2       1.59331       1.58132     0.829861  │
│ train_75aa1912   TERMINATED              40              1   leaky_relu               25           406.446              24       0.217741      0.230732    0.925694  │
│ train_5bba3053   TERMINATED             232              2   leaky_relu                3           692.811               2       1.55241       1.78088     0.832639  │
│ train_5dc01b02   TERMINATED             216              1   relu                      9          1582.66                8       0.417784      0.44774     0.822222  │
│ train_7d0ecd2f   TERMINATED              24              3   leaky_relu                3            51.3202              2       1.34012       1.22422     0.396528  │
│ train_9b89b141   TERMINATED             280              1   leaky_relu                3           828.945               2       1.58148       1.56359     0.830556  │
│ train_4f6aa660   TERMINATED              72              2   leaky_relu                3           117.261               2       1.15092       0.845168    0.65625   │
│ train_6fa21bb2   TERMINATED             120              1   relu                      3           202.234               2       0.643727      0.765281    0.529861  │
│ train_04503df6   TERMINATED             200              1   leaky_relu                3           474.739               2       0.724304      0.657753    0.877083  │
│ train_8fdb1884   TERMINATED             136              2   leaky_relu                3           280.472               2       1.47636       1.48563     0.786806  │
│ train_10e98fdf   TERMINATED             256              4   relu                      3           907.295               2       1.59625       1.58638     0.818056  │
│ train_4fcccca2   TERMINATED             128              1   leaky_relu                9           588.882               8       0.452394      0.520931    0.809722  │
│ train_7b87570c   TERMINATED              96              2   leaky_relu                3           142.692               2       1.44724       1.40886     0.565972  │
│ train_bd1a2b0a   TERMINATED             296              1   leaky_relu                9          2822.72                8       0.424507      0.495398    0.8375    │
│ train_7d0ca6d6   TERMINATED             168              1   gelu                     25          2961.77               24       0.231185      0.263104    0.938194  │
│ train_0a1a5642   TERMINATED             240              3   relu                      3           800.76                2       1.6159        1.60465     0.818056  │
│ train_1653f20d   TERMINATED               8              1   leaky_relu                3            18.8007              2       1.2285        1.13778     0.549306  │
│ train_0feded98   TERMINATED             264              2   gelu                      3           911.782               2       1.59686       1.54887     0.786111  │
│ train_9e35fd36   TERMINATED              40              1   leaky_relu                3            63.1741              2       0.803303      0.720094    0.752778  │
│ train_a12ea9c2   TERMINATED             272              1   gelu                      3           787.44                2       0.697185      0.84525     0.719444  │
│ train_30e38454   TERMINATED              56              1   leaky_relu                3            93.0447              2       0.793833      0.63551     0.791667  │
│ train_7d903f1d   TERMINATED             224              2   gelu                      3           616.243               2       1.50421       1.43477     0.761806  │
│ train_a0cbad8e   TERMINATED             184              3   relu                      3           520.49                2       1.52567       1.48821     0.726389  │
│ train_1c5b7f43   TERMINATED             288              1   leaky_relu                3           828.121               2       1.59125       1.59737     0.825     │
│ train_5c651a65   TERMINATED             112              2   gelu                      3           191.661               2       1.14058       1.05462     0.798611  │
│ train_c96243c9   TERMINATED             128              1   leaky_relu               25          1599.36               24       0.223476      0.314285    0.901389  │
│ train_75c47083   TERMINATED             248              1   gelu                     25          4489.66               24       0.310016      0.334961    0.927083  │
│ train_d64c75af   TERMINATED             200              4   leaky_relu                3           656.908               2       1.5786        1.56876     0.825694  │
│ train_c54d65e4   TERMINATED              32              1   relu                      9           102.907               8       0.443372      0.404812    0.895833  │
│ train_1bc6d9b2   TERMINATED              48              2   leaky_relu                3            61.7912              2       1.44787       1.3935      0.507639  │
│ train_87290a35   TERMINATED             144              1   gelu                      9           725.051               8       0.362615      0.495574    0.782639  │
│ train_2c5a42a4   TERMINATED             192              1   leaky_relu                9          1169.73                8       0.433509      0.471034    0.791667  │
│ train_2e4bb948   TERMINATED              16              2   gelu                      3            26.7233              2       1.1404        1.08397     0.539583  │
│ train_1651a5b9   TERMINATED              80              1   leaky_relu               25           822.111              24       0.213087      0.167632    0.93125   │
│ train_fa1e0603   TERMINATED             208              1   relu                      3           472.695               2       0.977984      0.777397    0.590278  │
│ train_93b0ccc3   TERMINATED             176              2   leaky_relu                3           398.718               2       1.58414       1.54023     0.711806  │
│ train_2421bb66   TERMINATED              40              3   gelu                      3            72.3665              2       1.50863       1.42228     0.209722  │
│ train_cb8eaabe   TERMINATED             160              1   leaky_relu                3           292.596               2       0.727687      0.737689    0.515972  │
│ train_2ab13d34   TERMINATED             152              1   gelu                      9           862.603               8       0.375968      0.481927    0.922917  │
│ train_7aadc501   TERMINATED              64              2   leaky_relu                3            84.1696              2       1.58389       1.54579     0.840278  │
│ train_67f1ffdb   TERMINATED              80              1   leaky_relu               25           824.997              24       0.181835      0.354225    0.952083  │
│ train_d149ecd8   TERMINATED              80              1   leaky_relu                3           101.149               2       0.609181      0.739772    0.844444  │
│ train_64d15a0b   TERMINATED             280              1   leaky_relu                9          2350.63                8       0.371857      0.462103    0.84375   │
│ train_bece87f2   TERMINATED             216              4   leaky_relu                3           742.75                2       1.57308       1.58444     0.838194  │
│ train_d26e37d9   TERMINATED              80              2   leaky_relu                3           116.468               2       1.59818       1.57142     0.0625    │
│ train_e4d921ef   TERMINATED              72              1   leaky_relu                9           293.961               8       0.36641       0.520211    0.854861  │
│ train_6941ae57   TERMINATED             200              1   leaky_relu               25          2811.61               24       0.18675       0.280362    0.922222  │
│ train_aaa21b4e   TERMINATED              88              3   leaky_relu                3           174.663               2       1.49401       1.39884     0.54375   │
│ train_76213aa7   TERMINATED             104              1   leaky_relu               25          1468.87               24       0.241258      0.232169    0.914583  │
│ train_9664fb49   TERMINATED             264              2   relu                      3           986.706               2       1.6107        1.60251     0.827778  │
│ train_b62e6e0d   TERMINATED             264              1   gelu                      3           859.304               2       0.724324      0.684946    0.654861  │
│ train_1af1f087   TERMINATED             264              2   gelu                      3           994.924               2       1.49552       1.46988     0.684722  │
│ train_52ad2318   TERMINATED              24              1   relu                      9           136.552               8       0.412078      0.427657    0.840972  │
│ train_7b32d8d9   TERMINATED             136              3   gelu                      3           368.852               2       1.56774       1.51734     0.125     │
│ train_f9a870e9   TERMINATED             120              1   leaky_relu               25          1504.09               24       0.221008      0.335769    0.870139  │
│ train_b3ce51c6   TERMINATED             232              1   leaky_relu                3           636.775               2       0.669852      0.648716    0.809722  │
│ train_ce57f550   TERMINATED             256              1   leaky_relu                9          1355.29                8       0.440888      0.428368    0.898611  │
│ train_5dd68233   TERMINATED              80              1   leaky_relu                9           335.711               8       0.389734      0.393662    0.843056  │
│ train_e84a2c7c   TERMINATED              96              1   leaky_relu               25           833.606              24       0.25882       0.247546    0.918056  │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯


│ Trial name       status         hidden_size     num_layers   dense_activation       iter     total time (s)     iterations     train_loss     test_loss     Accuracy │
│ train_1651a5b9   TERMINATED              80              1   leaky_relu               25           822.111              24       0.213087      0.167632    0.93125   │
Best result has 1 layer, hidden size <100. For the test with the recall visible i will decrease the hidden size range and the amount of layers and increase the 

"""