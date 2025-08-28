# within_host_inference

This repo supports our work on within-host inference. All codes are written in Julia and downloading the repo and initialising the environment
will allow you to run each script.

## Running models

The main models are located in the `src` folder. The files:

- `within_host_inference_run.jl` — runs the within-host model for the simulated dataset.
- `within_host_inference_nba_run.jl` — runs the within-host model for the NBA dataset.

At the moment these scripts can be run in the Julia REPL. They can be run in sections or line by line and will sample
from the posterior of the model, saving a dataframe of the results. 

## Analysis

Code for the analysis of the results is featured in `src/analysis` and the main scripts here are: 

- `results_posteriors.jl` — this script will generate the posteriors for the results of the within-host model.
- `results_posteriors_nba.jl` — this script will generate the posteriors for the results of the within-host model for the NBA dataset.
- `results_simulations.jl` — this script will generate posterior predictive VL trajectories for the within-host model for the simulated dataset.
- `results_simulations_nba.jl` — this script will generate posterior predictive VL trajectories for the within-host model for the NBA dataset.

## For easier navigation

### Data processing / generation