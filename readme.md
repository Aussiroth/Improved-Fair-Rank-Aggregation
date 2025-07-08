This is the repository for the paper 'Improved Fair Rank Aggregation Under Fairness Constraints'

The full paper can be found on arxiv [at this link](https://arxiv.org/abs/2505.10006).

## Requirements/Dependencies

The code was tested to work using the following versions.
- Python 3.12.
- Jupyterlab 4.1.5
- CVXPY 1.4.2
- PySCIPOpt 5.1.1

Other integer linear program solvers should also work, and may be faster.

## Code

The code can be found in both a Jupyter notebook, ``fairaggregation.ipynb`` as well as a standalone python script ``fairra.py``.

In particular, we have three implementations - The Best From Input [Chakraborty et al.], our paper's algorithm using Kwiksort as the fair rank aggregation subroutine (as a practical implementation), and our paper's algorithm using ILP as the subroutine (as the best-case scenario).

We also provide an ILP model that solves the fair rank aggregation optimally which can be used to find the optimal solution as a basis for comparison.

## Data

The ``football`` folder contains the data for the football dataset. Each instance of rank aggregation are stored within one CSV file each, ``week1.csv`` to ``week16.csv``. For each file, there are 25 lines. Each line represents an input ranking over the players.
The ``attributes.csv`` file contains the mapping for players to attributes (groups), for all of the instances.

The ``movielens`` folder contains the data for the movielens dataset. ``unique_200.txt`` contains the original dataset. ``movielens.txt`` is identical in data, but preprocessed to have similar structure as the football dataset for ease of use.
``movielens_reduced.txt`` is the reduced dataset with 58 movies, instead of the original 268. We use this as the input instance for experiments in the paper on the 'reduced movielens' dataset.
In each of the files, there are 7 lines. Each line represents an input ranking over the movies. The movies are represented by their ids.
The ``attributes.csv`` file contains the mapping for movies to groups, for the corresponding input instance.

Again we would like to thank the previous researchers for these datasets. The data for the football dataset is from Rank aggregation algorithms for fair consensus (Kuhlman and Rundensteiner 2020) and the data for the movielens dataset from Rank Aggregation with Proportionate Fairness (Wei et al. 2022).