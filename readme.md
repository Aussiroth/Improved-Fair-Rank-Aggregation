Folder structure and files:

fairaggregation.ipynb is the main notebook with code for replicating results. It contains
- Functions for reading in the data sources
- Implementation of an ILP solver for the optimal, our algorithm and the previous state of the art.
- Examples for how to read in an instance and solve it with each algorithm implementation

The football folder contains the data for the football dataset. Each of the instances are stored within one CSV file each, week1 to week16.
The attributes.csv file contains the mapping for players to groups, for all instances (that is, always use this file for the mapping, for all instances)

The movielens folder contains the data for the movielens dataset. unique_200.txt is the original dataset. movielens.txt is identical in data, but preprocessed to have similar structure as the football dataset.
movielens_reduced is the reduced dataset with 58 movies, instead of the original 268. We use this as the input instance for experiments in the paper on the 'reduced movielens' dataset.
The attributes.csv file again contains the mapping for movies to groups, for the corresponding input instance.
The Clean_Movielens.ipynb notebook contains the code that preprocessed the data from the raw form into the form used for input to our implementations.

Note that we have implemented Kwiksort as a subroutine for solving standard rank aggregation in our algorithm. It should be noted that it is a randomized algorithm.
To exactly reproduce our results, the numpy default rng generator should be seeded with a specific array of values.
The array should be [dataset, k, n, d, alpha_product] for the football dataset, and [k, n, d, alpha_product] for the movielens dataset.
alpha_product is equal to the product of alpha_0 to alpha_g, multiplied by 100 each (to get a large enough, hopefully unique, integer from this process)
This selection was to prevent the same seed from being used for each experiment when changing various parameters of the input.