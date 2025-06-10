# MADS-exam-25
build with [![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

```bash
├── README.md
├── data                            <- datafiles
│   ├── heart_big_test.parq
│   ├── heart_big_train.parq
│   ├── heart_test.parq
│   └── heart_train.parq
├── dev                             <- demo hypertune files to start developping and debugging
│   ├── Makefile
│   ├── hypertune.py                <- gestures hypertune file, copied from lesson 4
│   ├── minimal_ray.py              <- ray tune with minimal dependencies, from ray documentation
├── notebooks                       <- notebooks to get you started
│   ├── 01_explore-heart.ipynb
│   ├── 02_2D-models.ipynb
│   ├── 03_1D-models.ipynb
│   ├── Makefile                    <- makefile for mlflow on linux/osx
│   ├── config.toml
│   ├── start_server.sh             <- bash scripts to start/stop mlflow, if you dont have make
│   └── stop_server.sh
├── src                             <- src code
│   ├── __init__.py
│   ├── datasets.py                 <- datasets classes
│   ├── metrics.py                  <- metrics functions to handle torch/sklearn metrics
├── pyproject.toml                  <- dependencies
└── uv.lock
```

## how to set this up
If you are on a VM, first check your available space with

```bash
df -h
```

If your disk is full, likely candidates are the `.venv` folders of your projects, or the `.cache` folder in your home directory, or specifically the uv cache.

1. clean up `.venv` folders by cd-ing into the project folder and running

```bash
rm -rf .venv
```

2. clean up the uv cache folder by running

```bash
uv cache clean
```

If you have less than 7GB of space, you really need to free up some space.
You can check size of (sub)folders with, eg cd to `~/.cache` and run

```bash
du -sh *
```

To pull the datafiles, you need to install [git-lfs](https://git-lfs.com), see link for other OS than linux:

```bash
sudo apt install git-lfs
```

If you have sufficient space, do this:
1. clone the repo
2. Run `git lfs pull` to get the datafiles
3. create your own repo on github, copy your-repo-url
4. in the cloned repo, change the remote url with `git remote set-url origin <your-repo-url>` such that you can push to your-repo-url
5. use the `pyproject.toml` to install the dependencies.

Please note that the pyproject.toml file specifies the torch cpu version to save memory. If you are on a machine with GPU, you can remove this and it will download GPU versions.


## The case
The junior datascientist at your work is pretty confident about his knowledge of all the models; He has helped you out by doing some data exploration for you, he engineered a custom dataset for you, and he has found some models that at least did something. You can find all of his work in the `notebooks` folder.

He didn't learn to hypertune models, and he really wouldnt dare to modify architectures.

However, you have been bragging about your high grades on the ML course. In addition to that, you have been telling weird stories about how vision models are the same as searching for undigested owl pellets in the forest, and while nobody really understood what you said, they are really under the impression that if someone can do it, it should be you.

so now you are in the situation to actually show your skills; hypertune the models, and show your creativity by improving the models, and maybe even come up with some new approaches.

## The data
We have two datasets:
### The PTB Diagnostic ECG Database

- Number of Samples: 14552
- Number of Categories: 2
- Sampling Frequency: 125Hz
- Data Source: Physionet's PTB Diagnostic Database

All the samples are cropped, downsampled and padded with zeroes if necessary to the fixed dimension of 187. There is a target column named "target".

You can find this in `data/heart_train.parq` and `data/heart_test.parq`.

### Arrhythmia Dataset

- Number of Samples: 109446
- Number of Categories: 5
- Sampling Frequency: 125Hz
- Data Source: Physionet's MIT-BIH Arrhythmia Dataset
- Classes: ['N': 0, 'S': 1, 'V': 2, 'F': 3, 'Q': 4]
All the samples are cropped, downsampled and padded with zeroes if necessary to the fixed dimension of 187. There is a target column named "target".

You can find this in `data/heart_big_train.parq` and `data/heart_big_test.parq`.

## Analysis
In `notebooks/01_explore-heart.ipynb` you can find some visualisations. It's not much, and
he hasnt had time to add a lot of comments, but he thinks it's pretty self explanatory for
someone with your skill level.

## Models
There are two notebooks with models. He has tried two approaches: a 2D approach, and a 1D approach. Yes, he is aware that this is a time series, but he has read that 2D approaches can work well for timeseries, and he has tried it out.

# Your task
Your task is to create a model for the Arrhythmia dataset. The PTB dataset might be used to test some architectures, or maybe to pretrain a model, but it isnt mandatory to use. During the process, you will be going through an iterative scientific process:
1. Create a hypothesis
2. Create an experiment to test your hypothesis
3. Reflect on the results
4. Go back to 1 to refine your hypothesis

During the process, it is advised you keep track of your ideas, hypotheses and expectations in a journal; think of this as a logbook with short notes and timestamps. A suggestion is to use https://jrnl.sh/en/stable/ for fast note taking directly from the cli, but you can essentially use anything, even pen and paper.

# 1. Explore
First, have a look at the notebooks. Then, take a step back and try to make an inventory of all possible solutions:
- What models might work here?
- What are ranges of hyperparameters you think are reasonable?

Change and expand the existing models: add more parameters you might want to tune, add additional layers, etc.
In addition to that, think about other strategies beyond the two models. Try to balance exploit (improve what is already there) and explore (creativity/curiosity of new things).

Try to formulate your hypothesis: what do you think will work best, and why? Connect your ideas to theory.
Then, try to set up some experiments to test your ideas. Implement them in a flexible way, so you can easily do experiments at scale.
Test, and reflect on the results by looking at your hypotheses; what surprised you, and most important, WHY did it surprise you?
Do some manual hypertuing to get a feel for the models and what might work, or what does not.

Because this is a medical dataset, an we are trying to spot disease, recall is more important than precision. Make sure you take this into account when you are creating and evaluating models! This means you might need to be creative about the preprocessing, or maybe combine approaches, or dive into sampling methods like [SMOTE](https://imbalanced-learn.org/dev/references/generated/imblearn.over_sampling.SMOTE.html).

Be smart with the number of epochs. If you are doing manual tests, you can lower the amount of epochs; your goal is to see differences.
However, you will probably need some more training than just 1 epoch (which is equivalent with 5 epochs when trainsteps is set to one fifth of the full dataset), especially for the underrepresented classes. So try to set up experiments that can run unattended for a longer time without the need of your constant supervision.

*NOTE*: separate your own models from existing code (eg from `mltrainer`). I want to be able to see what YOUR skill-level is.

## 2 Hypertune
### 2.1 Select
Once you have some idea of what works, and what doesnt, you can make a selection of interesting directions.
Make your selection based on your hypotheses of what might work, and what you have seen in the manual hypertuning.
### 2.2 Tune
Set up your own `hypertune.py` file and hypertune the models with Ray. Make sure you log everything thats relevant for your model (eg with ray / mlflow). Mark your own additions and modifications with comments.
## 2.3 Analyse
Make your hypertuning iterative; you have an hypothesis, do an experiment, think about what it means, log those ideas, change the searchspace range based on what you saw, and set up a new experiment. Make smart use of approaches like gridsearch, randomsearch, bayesian optimization, hyperband, etc.

Make sure you save all plots you might need for your report.
## 3. Reflect
Create a summary (max 3 A4 pages of text) of your results, excluding the title page and figures/tables.
1. Start with an overview of your top architectures. Make sure someone can reproduce your model based on your description.
2. Describe your searchspace, and show the relevant hyperparameters you have tuned. You can take inspiration from table 3 in the (attention is all you need)[https://arxiv.org/pdf/1706.03762.pdf] paper for an excellent summary of hypertuning.
3. Connect your results to your hypotheses. What did you discover? What are new insights? What hypotheses were confirmed, and which were rejected?

Export your report to pdf (you can use latex, which will also be useful for your master theses. This is a useful template https://github.com/raoulg/paper-template if you are interested)

When you are done, do two things:
1. Make your own repo and invite https://github.com/raoulg to your repo as a collaborator
2. push all your code, and **add the pdf** to the repo.

# Rubric
You will be graded for:
- Overall presentation and clarity of your work (organisation, comments, typehinting, etc). (10%)
- The level of change and expansion of the models: balance performance (exploit) and creativity/curiosity (explore). (30%)
- The quality of your implementation of the hypertuning in your codebase (20%)
- The clarity and relevance of your reflection on the hyperparameters, models and hypotheses (40%)

Note I DO NOT grade your accuracy; I will grade the quality of the process and reflection.

## Checklist
- [ ] you have a system in place for logging your exploration process and you are reminded automatically to log hypotheses
- [ ] you **explored** different models, hyperparameter ranges, layers, combinations, etc.
- [ ] you have **connected** your practical ideas to the theory before the experiments: why might this work?
- [ ] you show you worked in an **iterative** process: hypothesis -> experiment -> reflection -> new hypothesis
- [ ] after your results, you reflected on your hypotheses and refined them, when necessary
- [ ] you have a clear overview and **representation** of your top architectures in your report (eg in a visual block scheme of layers and connections), such that someone can easily understand and reproduce your model
- [ ] you have a clear overview of the (most relevant) **searchspace** in your report
- [ ] you have connected your results to your hypotheses in your reflection
- [ ] you have a pdf of your report, that has the **URL of the repo** on the first page
- [ ] you have used **linters** (isort, ruff, pyright/mypy) on your code
- [ ] you have a repo with a clear README that contains your name & student number, and instructions how to find and read your code
- [ ] you have invited raoulg to your repo

