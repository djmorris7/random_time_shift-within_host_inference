# within_host_inference

This repository contatins the code to run the within-host inference for viral dynamics models using the random time-shift approximation. This supports our paper titled *Random time-shift approximation enables hierarchical Bayesian inference of mechanistic within-host viral dynamics models on large datasets*. The data is provided in the `data` folder.

## Setup and installation

To run the code in this repository, you will need to have Julia installed on your machine. You can download it from [the official Julia website](https://julialang.org/downloads/).
Once you have Julia installed, you can clone this repository and set up the environment by following these steps:

1. Clone the repository.
2. Activate the julia environment. This will start Julia with the project environment set to the current directory. In the Julia REPL, you can use the command `] activate .` in the Julia REPL. This will activate the environment defined in the `Project.toml` and `Manifest.toml` files in the current directory. If you are using Visual Studio Code, you can also open the Command Palette (Cmd+Shift+P on macOS or Ctrl+Shift+P on Windows/Linux) and type "Julia: Activate Environment" to activate the environment. If you press `]`, you should see a prompt like `(within_host_inference) pkg>`, indicating that you are in the correct environment.
3. Instantiate the environment. To do this, run the command `] instantiate` in the Julia REPL. This will read the `Project.toml` and `Manifest.toml` files in the current directory and install all the required packages. This should install all the required packages specified in the `Project.toml` and `Manifest.toml` files.

## Tips for running the code

We recommend using the [Visual Studio Code](https://code.visualstudio.com/) editor with the [Julia extension](https://marketplace.visualstudio.com/items?itemName=julialang.language-julia) for a better development experience.
In the following steps, unless otherwise specified, you should run the scripts in sections, i.e. in the **notebook** style way. This can be done by using `cmd+shift+enter` in VS Code.

### 1. Neural network

**This code must be run to create the Neural network which is needed for running the inference.**

1. In the `src/nn` folder, you will find the code to generate the data for the neural network model. First run the script `nn_data_generation_run.jl` to generate the data. This will populate the `data/tcl_nn` folder with the data files. 
2. Next, run the script `train_nn.jl` to train the neural network model. This will save the trained model in the `data/tcl_nn` folder.

### 2. Data generation and inference for synthetic dataset

**Note: Steps 1 and 2 should not need to be run as the data already exist in the data folder.**

1. To generate the synthetic dataset, you will need to run the script `src/generate_data_R0.jl`. This script will generate a synthetic dataset of viral load trajectories and save it in the `data` folder. 
2. Next, run the script `src/inference/process_sim_data.jl` to process the data. 
3. Now we can run the inference on the synthetic dataset. Run the script `src/within_host_inference_run.jl`. This will run the within-host model on the synthetic dataset and save the results in a csv in the `results` folder.

### 3. Data processing and inference for NBA dataset

**Note: Steps 1 and 2 should not need to be run as the data already exist in the data folder.**

1. The NBA dataset will be available in the `data/nba` folder. If you want to use a different dataset, you can replace the files in this folder with your own data.
2. We run `src/inference/process_nba_data.jl` script to process the NBA dataset. This will save the processed data in the `data/nba` folder.
3. Now we can run the inference on the NBA dataset. Run the script `src/within_host_inference_nba_run.jl`. This will run the within-host model on the NBA dataset and save the results in a csv in the `results/nba` folder.

### Analysis of results

- `src/analysis/intro_fig.jl` — this script will generate the introductory figure for the paper.
- `src/analysis/plot_data.jl` — this script will generate the data plot. 
- `src/analysis/results_posteriors_nba.jl` — this script will generate the posteriors for the results of the within-host model for the NBA dataset.
- `src/analysis/results_posteriors.jl` — this script will generate the posteriors for the results of the within-host model for the synthetic dataset.
- `src/analysis/results_simulations.jl` — this script will generate posterior predictive VL trajectories for the within-host model for the synthetic dataset.
- `src/analysis/results_simulations_nba.jl` — this script will generate posterior predictive VL trajectories for the within-host model for the NBA dataset.