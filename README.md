# Getting started for PRA_PLHS

To be able to follow the exercises, you are going to need a laptop with Anaconda and several Python packages installed.
The following instruction would work as is for Mac or Ubuntu Linux users, Windows users would need to install and work in the [Git BASH](https://gitforwindows.org/) terminal.


## Download and install Anaconda

Please go to the [Anaconda website](https://www.anaconda.com/).
Download and install *the latest* Anaconda version for latest *Python* for your operating system.


## Check-out the git repository of PRA_PLHS

Once Miniconda is ready, checkout the course repository and proceed with setting up the environment:

```bash
git clone https://github.com/seungsab/PRA_PLHS.git
```


## Create isolated Anaconda environment

Just type:

```bash
conda env create -f environment.yml
source activate PRA_PLHS
```
[Documents for VARS-TOOL](https://vars-tool.readthedocs.io/en/latest/index.html)
