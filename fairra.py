import cvxpy as cp
import numpy as np
import random
import math
import time
from collections import deque

print(cp.installed_solvers())

####
####
#Helper functions
####
####

def Kendall_Tau_Dist(first, second):
    mappedrank = []
    for i in range(len(second)):
        mappedrank.append(first.index(second[i]))
    cost, blank = mergesort(mappedrank)
    return cost

#mergesort to compute distance in nlogn time
#input: A single ranking
#output: Kendall tau distance to the ranking 1, 2, ..., n
def mergesort(ranking):
    if len(ranking) <= 1:
        return 0, ranking
    leftsum, leftrank = mergesort(ranking[:len(ranking)//2])
    rightsum, rightrank = mergesort(ranking[len(ranking)//2:])
    csum = leftsum + rightsum
    leftindex = 0
    rightindex = 0
    outrank = []
    while leftindex < len(leftrank) and rightindex < len(rightrank):
        if leftrank[leftindex] < rightrank[rightindex]:
            outrank.append(leftrank[leftindex])
            leftindex += 1
        else:
            outrank.append(rightrank[rightindex])
            csum += len(leftrank) - leftindex
            rightindex += 1
    if leftindex < len(leftrank):
        outrank += leftrank[leftindex:]
    if rightindex < len(rightrank):
        outrank += rightrank[rightindex:]
    return csum, outrank
    
def Get_Objective_Value(query, rankings):
    median_cost = 0
    for rank in rankings:
        median_cost += Kendall_Tau_Dist(query, rank)
    return median_cost

#helper function to return the weighted tournament corresponding to the rank aggregation problem
def Get_Frac_Tournament(rankings):
    element_count = len(rankings[0])
    frac_tournament = np.ndarray((element_count, element_count))
    for i in range(element_count):
        for j in range(element_count):
            frac_tournament[i][j] = 0

    for ranking in rankings:
        for i in range(len(ranking)):
            for j in range(i+1, len(ranking)):
                frac_tournament[ranking[i]][ranking[j]] += 1
    for i in range(element_count):
        for j in range(element_count):
            frac_tournament[i][j] = frac_tournament[i][j] / len(rankings)

    return frac_tournament
    
#helper function to recover ordering from acyclic tournament
def Topological_Sort(adj):
    n = len(adj)  
    in_degree = [0] * n

    for i in range(n):
        for j in range(n):
            if adj[i][j] > 0.5:
                in_degree[j] += 1

    queue = deque()
    for i in range(n):
        if in_degree[i] == 0:
            queue.append(i)

    topo_sort = []

    while queue:
        node = queue.popleft()
        topo_sort.append(node)

        for j in range(n):
            if adj[node][j] > 0.5:
                in_degree[j] -= 1
                if in_degree[j] == 0:
                    queue.append(j)

    return topo_sort
    
####
####
# End of helper functions
####

    
#Input: set of rankings
#Returns: the optimal median ranking by using an ILP
#See "Improved Bounds for Computing Kemeny Rankings" for related information.
def NormalILP(rankings):
    element_count = len(rankings[0])
    
    frac_tournament = Get_Frac_Tournament(rankings)
    
    X = cp.Variable(element_count * element_count, boolean = True)
    constraints = []
    
    #constraints that for every pair, one is before the other
    for i in range(element_count):
        for j in range(element_count):
            coeff = np.zeros(element_count * element_count)
            coeff[i*element_count + j] = 1
            coeff[j*element_count + i] = 1
            constraints += [coeff @ X == 1]
    
    #triangle inequality constraint
    #x_ab + x_bc + x_ca >= 1 for any a, b, c
    for i in range(element_count):
        for j in range(element_count):
            if i == j:
                continue
            for k in range(element_count):
                if i == k or j == k:
                    continue
                coeff = np.zeros(element_count * element_count)
                coeff[i*element_count + j] = 1
                coeff[j*element_count + k] = 1
                coeff[k*element_count + i] = 1
                constraints += [coeff @ X >= 1]
    
    edge_weight_coeff = np.empty(element_count * element_count)
    for i in range(element_count):
        for j in range(element_count):
            edge_weight_coeff[i * element_count + j] = frac_tournament[i][j]
            
    problem = cp.Problem(cp.Minimize(edge_weight_coeff @ X), constraints)
    
    print("Constraints done. Solving...")
    problem.solve(solver = cp.SCIP)

    result = X.value.reshape(element_count, -1)
    for i in range(element_count):
        result[i][i] = 0
    result_tp = [[0]*element_count for i in range(element_count)]
    for i in range(element_count):
        for j in range(element_count):
            if i != j:
                result_tp[i][j] = result[j][i]
    
    topo_sorted = Topological_Sort(result_tp)
    print("Objective value: ", Get_Objective_Value(topo_sorted, rankings))
    return topo_sorted

#FAIR ILP
#Takes in the fairness parameters, rankings, mapping of elements to attributes
#Returns the optimal fair median ranking
def FairILP(alphas, betas, rankings, id_attribute, num_attributes):

    start_time = time.time()
    element_count = len(rankings[0])

    frac_tournament = Get_Frac_Tournament(rankings)

    #Large constant, must be bigger than 2 * elements
    bigM = element_count * 10
    
    X = cp.Variable(element_count * element_count, boolean = True)
    constraints = []
    
    #constraints that for every pair, one is before the other
    for i in range(element_count):
        for j in range(element_count):
            coeff = np.zeros(element_count * element_count)
            coeff[i*element_count + j] = 1
            coeff[j*element_count + i] = 1
            constraints += [coeff @ X == 1]
    
    #triangle inequality constraint
    #x_ab + x_bc + x_ca >= 1 for any a, b, c
    for i in range(element_count):
        for j in range(element_count):
            if i == j:
                continue
            for k in range(element_count):
                if i == k or j == k:
                    continue
                coeff = np.zeros(element_count * element_count)
                coeff[i*element_count + j] = 1
                coeff[j*element_count + k] = 1
                coeff[k*element_count + i] = 1
                constraints += [coeff @ X >= 1]

    #Y_a variables to enforce fairness
    #The pair of constraints force it so that Y = 1 if for some i, at least d - 1 - K X_ij are 0
    #Otherwise it is forced to be 0
    Y = cp.Variable(element_count, boolean = True)

    #to be in top-K, the element must be ordered ahead of at least d - K elements.
    largerthan_k = element_count - TOPK - 1
    for i in range(element_count):
        coeff = np.zeros(element_count * element_count)
        for j in range(element_count):
            if i != j:
                coeff[i * element_count + j] = -1
        constraints += [coeff @ X + element_count - 1 >= largerthan_k + 1 - bigM*(1 - Y[i])]
        constraints += [coeff @ X + element_count - 1 <= largerthan_k + bigM * Y[i]]

    #Lower and upper bound constraints per attribute
    for attribute in range(num_attributes):
        coeff = np.zeros(element_count)
        for i in range(element_count):
            if id_attribute[i] == attribute:
                coeff[i] = 1
        lb = math.floor(alphas[attribute] * TOPK)
        ub = math.ceil(betas[attribute] * TOPK)
        constraints += [coeff @ Y >= lb]
        constraints += [coeff @ Y <= ub]
    
    edge_weight_coeff = np.empty(element_count * element_count)
    for i in range(element_count):
        for j in range(element_count):
            edge_weight_coeff[i * element_count + j] = frac_tournament[i][j]
            
    problem = cp.Problem(cp.Minimize(edge_weight_coeff.T @ X), constraints)
    
    print("Constraints done. Solving...")
    problem.solve(solver = cp.SCIP)

    #use topological sorting algo to also get the ordering of elements
    result = X.value.reshape(element_count, -1)
    for i in range(element_count):
        result[i][i] = 0
    result_tp = [[0]*element_count for i in range(element_count)]
    for i in range(element_count):
        for j in range(element_count):
            if i != j:
                result_tp[i][j] = result[j][i]
    
    topo_sorted = Topological_Sort(result_tp)
    print("Resulting ranking: ", topo_sorted)
    obj_cost = Get_Objective_Value(topo_sorted, rankings)
    print("Objective value: ", obj_cost)
    end_time = time.time()
    print("Fair ILP took time " + str(end_time - start_time) + " seconds.")
    return obj_cost

#This implementation of our algorithm uses ILP to solve the two partitions optimally
#Think of this as the 'best case' scenario possible.
#Takes in the fairness parameters, rankings, mapping of elements to attributes
def OurAlgo(alphas, betas, rankings, id_attribute, num_attributes):

    element_count = len(rankings[0])

    #STEP 1: determining top-k elements
    #Construct weighted tournament, and then sort by indegrees, and take it as following the algorithm in the paper
   
    start_time = time.time()
    frac_tournament = Get_Frac_Tournament(rankings)

    fract_time = time.time()

    #List of lists.
    #List i contains tuples of elements with attribute i
    #tuple is in the form (element id, indegree)
    indegree_attr = []
    
    for attribute in range(num_attributes):
        indegree_attr.append([])
    for i in range(element_count):
        i_attr = id_attribute[i]
        indeg = 0
        for j in range(element_count):
            indeg += frac_tournament[j][i]
        indegree_attr[i_attr].append((i, indeg))
    for attr in range(num_attributes):
        indegree_attr[attr].sort(key = lambda ituple : ituple[1])

    topk_elements = set()
    elements_taken = [0] * num_attributes
    num_taken = 0
    
    #now, we get top k elements following the algo
    #take lower bound first
    #form combined list at same time
    indegree_combined = []
    for attr in range(num_attributes):
        for j in range(math.floor(alphas[attr] * TOPK)):
            topk_elements.add(indegree_attr[attr][j][0])
            elements_taken[attr] += 1
        indegree_combined += indegree_attr[attr][math.floor(alphas[attr] * TOPK):]
    
    #sort combined list, then take while respecting beta upper bounds
    indegree_combined.sort(key = lambda ituple : ituple[1])
    for i in range(len(indegree_combined)):
        if len(topk_elements) >= TOPK:
            break
        element = indegree_combined[i]
        i_attr = id_attribute[element[0]]
        if elements_taken[i_attr] < math.ceil(betas[i_attr] * TOPK):
            elements_taken[i_attr] += 1
            topk_elements.add(element[0])


    #STEP 2, we need to order the top-k.
    #Following the paper, we need to run rank aggregation over the two partitions.
    
    #In this implementation, ILP is used to solve optimally, giving the best case scenario.
    
    #So we need to construct the restricted rankings
    #left is top k, right is the remaining elements
    
    rankings_left = []
    rankings_right = []
    for rank in rankings:
        left_rank = []
        right_rank = []
        for i in rank:
            if i in topk_elements:
                left_rank.append(i)
            else:
                right_rank.append(i)
        rankings_left.append(left_rank)
        rankings_right.append(right_rank)

    #NOTE: Because the elements of the reduced rankings are not a continuous 1 ... k, we need to relabel the elements to be 1 ... k, and save the mapping
    #so we can map the result back to these elements

    left_forward_map = {}
    left_backward_map = {}
    mapped_rankings_left = []
    for i in range(len(rankings_left[0])):
        left_forward_map[rankings_left[0][i]] = i
        left_backward_map[i] = rankings_left[0][i]
    mapped_rankings_left.append([i for i in range(len(rankings_left[0]))])
    for i in range(1, len(rankings_left)):
        mapped_rank = []
        for j in rankings_left[i]:
            mapped_rank.append(left_forward_map[j])
        mapped_rankings_left.append(mapped_rank)

    right_forward_map = {}
    right_backward_map = {}
    mapped_rankings_right = []
    for i in range(len(rankings_right[0])):
        right_forward_map[rankings_right[0][i]] = i
        right_backward_map[i] = rankings_right[0][i]
    mapped_rankings_right.append([i for i in range(len(rankings_right[0]))])
    for i in range(1, len(rankings_right)):
        mapped_rank = []
        for j in rankings_right[i]:
            mapped_rank.append(right_forward_map[j])
        mapped_rankings_right.append(mapped_rank)
    
    #use ILP to solve
    left_topo_sorted = NormalILP(mapped_rankings_left)
    right_topo_sorted = NormalILP(mapped_rankings_right)
    
    #Re-map the topologically sorted elements, to the original elements using the backward maps

    left_original = [left_backward_map[i] for i in left_topo_sorted]
    right_original = [right_backward_map[i] for i in right_topo_sorted]

    output_ranking = left_original + right_original

    #Get objective cost
    obj_cost = Get_Objective_Value(output_ranking, rankings)
    print("Objective cost of our ranking is: ", obj_cost)

    end_time = time.time()
    print("Algo took time " + str(end_time - start_time) + " seconds.")
    return obj_cost

    #print("Our algorithm ranking is: ", output_ranking)

#This implementation of our algorithm uses KWIKSORT to solve the standard rank aggregation problem
#For details on KWIKSORT, see Ailon, Newman, Charikar 2007
def OurAlgo_KS(alphas, betas, rankings, id_attribute, num_attributes):
    element_count = len(rankings[0])

    #STEP 1: determining top-k elements
    #Construct weighted tournament, and then sort by indegrees, and take it as following the algorithm in the paper

    start_time = time.time()
    frac_tournament = Get_Frac_Tournament(rankings)

    fract_time = time.time()

    #List of lists.
    #List i contains tuples of elements with attribute i
    #tuple is in the form (element id, indegree)
    indegree_attr = []
    
    for attribute in range(num_attributes):
        indegree_attr.append([])
    for i in range(element_count):
        i_attr = id_attribute[i]
        indeg = 0
        for j in range(element_count):
            indeg += frac_tournament[j][i]
        indegree_attr[i_attr].append((i, indeg))
    for attr in range(num_attributes):
        indegree_attr[attr].sort(key = lambda ituple : ituple[1])
        
    topk_elements = set()
    elements_taken = [0] * num_attributes
    num_taken = 0
    #now, we get top k elements following the algo
    #take lower bound first
    #form combined list at same time
    indegree_combined = []
    for attr in range(num_attributes):
        for j in range(math.floor(alphas[attr] * TOPK)):
            topk_elements.add(indegree_attr[attr][j][0])
            elements_taken[attr] += 1
        indegree_combined += indegree_attr[attr][math.floor(alphas[attr] * TOPK):]
    
    #sort combined list, then take while respecting beta upper bounds
    indegree_combined.sort(key = lambda ituple : ituple[1])
    for i in range(len(indegree_combined)):
        if len(topk_elements) >= TOPK:
            break
        element = indegree_combined[i]
        i_attr = id_attribute[element[0]]
        if elements_taken[i_attr] < math.ceil(betas[i_attr] * TOPK):
            elements_taken[i_attr] += 1
            topk_elements.add(element[0])


    #STEP 2, we need to order the top-k.
    #Following the paper, we need to run rank aggregation over the two partitions.
    
    #In this implementation, Kwiksort is used to solve approximately, runs fast and easy to implement

    #left is top k, the front part
    rankings_left = []
    rankings_right = []
    for rank in rankings:
        left_rank = []
        right_rank = []
        for i in rank:
            if i in topk_elements:
                left_rank.append(i)
            else:
                right_rank.append(i)
        rankings_left.append(left_rank)
        rankings_right.append(right_rank)

    #NOTE: Because the elements of the reduced rankings are not a continuous 1 ... k, we need to relabel the elements to be 1 ... k, and save the mapping
    #so we can map the result back to these elements

    left_forward_map = {}
    left_backward_map = {}
    mapped_rankings_left = []
    for i in range(len(rankings_left[0])):
        left_forward_map[rankings_left[0][i]] = i
        left_backward_map[i] = rankings_left[0][i]
    mapped_rankings_left.append([i for i in range(len(rankings_left[0]))])
    for i in range(1, len(rankings_left)):
        mapped_rank = []
        for j in rankings_left[i]:
            mapped_rank.append(left_forward_map[j])
        mapped_rankings_left.append(mapped_rank)

    right_forward_map = {}
    right_backward_map = {}
    mapped_rankings_right = []
    for i in range(len(rankings_right[0])):
        right_forward_map[rankings_right[0][i]] = i
        right_backward_map[i] = rankings_right[0][i]
    mapped_rankings_right.append([i for i in range(len(rankings_right[0]))])
    for i in range(1, len(rankings_right)):
        mapped_rank = []
        for j in rankings_right[i]:
            mapped_rank.append(right_forward_map[j])
        mapped_rankings_right.append(mapped_rank)

    #Using the better of kwiksort and best from input algorithms from Ailon, Newman, Charikar 2007 paper
    leftKwiksort = Kwiksort(mapped_rankings_left)
    rightKwiksort = Kwiksort(mapped_rankings_right)
    leftInput = Best_From_Input(mapped_rankings_left)
    rightInput = Best_From_Input(mapped_rankings_right)

    if Get_Objective_Value(leftKwiksort, mapped_rankings_left) < Get_Objective_Value(leftInput, mapped_rankings_left):
        left_topo_sorted = leftKwiksort
    else:
        left_topo_sorted = leftInput

    if Get_Objective_Value(rightKwiksort, mapped_rankings_right) < Get_Objective_Value(rightInput, mapped_rankings_right):
        right_topo_sorted = rightKwiksort
    else:
        right_topo_sorted = rightInput
    
    #Re-map the topologically sorted elements, to the original elements using the backward maps

    left_original = [left_backward_map[i] for i in left_topo_sorted]
    right_original = [right_backward_map[i] for i in right_topo_sorted]

    output_ranking = left_original + right_original

    #Get objective cost
    obj_cost = Get_Objective_Value(output_ranking, rankings)
    print("Objective cost of our ranking is: ", obj_cost)

    end_time = time.time()
    print("Algo (with kwiksort) took time " + str(end_time - start_time) + " seconds.")
    #print("Our algorithm ranking (with kwiksort) is: ", output_ranking)

    return obj_cost

##Helper functions for KWIKSORT
##Need to take bettter of best from input, and the kwiksort algorithm
def Best_From_Input(rankings):
    best_rank = []
    obj_value = 1e9
    for rank in rankings:
        median_cost = Get_Objective_Value(rank, rankings)
        if median_cost < obj_value:
            obj_value = median_cost
            best_rank = rank
    return best_rank

def Kwiksort(rankings):
    frac_tournament = Get_Frac_Tournament(rankings)
    initial = [i for i in range(len(rankings[0]))]
    rank = DoKwiksort(initial, frac_tournament)
    return rank

def DoKwiksort(elements, frac_tournament):
    if len(elements) <= 1:
        return elements
    pivot = rng.choice(elements)

    left = []
    right = []
    for element in elements:
        if element != pivot:
            if frac_tournament[element][pivot] >= 0.5:
                left.append(element)
            else:
                right.append(element)
    return DoKwiksort(left, frac_tournament) + [pivot] + DoKwiksort(right, frac_tournament)

#This algorithm is the 3-approximation best from input
#From Chakraborty et al. 2022
def BFI_Algo(alphas, betas, rankings, id_attribute, num_attributes):
    element_count = len(rankings[0])

    start_time = time.time()
    #Get closest fair ranking to each input rankings
    fair_rankings = []
    for rank in rankings:
        fair_rankings.append(Closest_Fair_Ranking(rank, id_attribute, num_attributes))

    #Pick the fair ranking which minimizes the objective value
    obj_value = 1e9
    best_rank = []
    for fair_rank in fair_rankings:
        median_cost = Get_Objective_Value(fair_rank, rankings)
        if median_cost < obj_value:
            obj_value = median_cost
            best_rank = fair_rank
    print("Best From Input algo objective value is: ", obj_value)
    #print("Best From Input algo rank is: ", best_rank)
    print("Best From Input took time " + str(time.time() - start_time) + " seconds.")
    return obj_value

#Helper function to find the closest fair ranking to the given rank
def Closest_Fair_Ranking(rank, id_attribute, num_attributes):
    elements_taken = [0] * num_attributes
    fair_rank = []
    topk_elements = set()
    for element in rank:
        if len(topk_elements) >= TOPK:
            break
        attr = id_attribute[element]
        if elements_taken[attr] < math.floor(alphas[attr] * TOPK):
            topk_elements.add(element)
            elements_taken[attr] += 1
    for element in rank:
        if len(topk_elements) >= TOPK:
            break
        if element not in topk_elements:
            if elements_taken[attr] < math.ceil(betas[attr] * TOPK):
                topk_elements.add(element)
                elements_taken[attr] += 1
    rear_part = []
    for element in rank:
        if element in topk_elements:
            fair_rank.append(element)
        else:
            rear_part.append(element)
    fair_rank += rear_part
    return fair_rank

####
####
# Data read functions
####
####

#This function is to read an instance of football dataset given the filename as input.
#Input: File name to read
#Returns: list of lists containing rankings; dictionary mapping elements to attributes; number of attributes; number of elements for each attribute
def Get_Football(fname):
    f = open(fname, "r")

    instance_players = f.readline().strip().split(",")
    f.seek(0)
    #need to read in attribute for fairness
    #these 2 lists will keep track of which group the player is in
    name_to_attribute = {}
    attributes = [[], []]
    attribute_f = open(r"football\attributes.csv", "r")
    for line in attribute_f:
        player = line.rstrip().split(",")
        player[1] = int(player[1])
        name_to_attribute[player[0]] = player[1]
        if player[0] in instance_players:
            attributes[player[1]].append(player[0])
    
    #now read in the input rankings, keeping only those who are in the set keep_players, else discard them from ranking
    player_toid = {}
    id_attribute = {}
    player_count = 0
    #list of input rankings
    rankings = []
    for line in f:
        rawrank = line.rstrip().split(",")
        rank = []
        for player in rawrank:
            if player not in player_toid:
                player_toid[player] = player_count
                id_attribute[player_count] = name_to_attribute[player]
                player_count += 1
            rank.append(player_toid[player])
        rankings.append(rank)

    attribute_count = {0: 0, 1: 0}
    for player in rankings[0]:
        attribute_count[id_attribute[player]] += 1
    
    return rankings, id_attribute, 2, attribute_count

    
#This function is to read an instance of movielens dataset.
#Returns: list of lists containing rankings; dictionary mapping elements to attributes; number of attributes; number of elements for each attribute
def Get_Movielens():
    #change file names here as desired
    f = open(r"Movielens\movielens_reduced.txt")
    attrf = open(r"Movielens\attributes_reduced.txt")

    movie_to_attribute = {}
    aid = 0
    attribute_id = {}
    attribute_counts = {}
    num_movies = 0
    for line in attrf:
        num_movies += 1
        line = line.rstrip().split(",")
        movie = line[0]
        if line[1] not in attribute_id:
            attribute_id[line[1]] = aid
            aid += 1
        attr = attribute_id[line[1]]
        movie_to_attribute[movie] = attr
        if attr in attribute_counts:
            attribute_counts[attr].append(movie)
        else:
            attribute_counts[attr] = [movie]
    
    rankings = []
    movie_to_id = {}
    id_to_attribute = {}
    counter = 0
    for line in f:
        ranking = []
        line = line.rstrip().split(",")
        for movie in line:
            if movie not in movie_to_id:
                movie_to_id[movie] = counter
                counter += 1
            mid = movie_to_id[movie]
            ranking.append(mid)
            id_to_attribute[mid] = movie_to_attribute[movie]
                
        rankings.append(ranking)

    return rankings, id_to_attribute, aid, attribute_counts
    
####
####
# End of data read functions
####

####
#Setup to run algorithms here

#The code expects that the input rankings are over the elements 0 ... d-1.
#If it is not 0 indexed, it will give an error.

#This shows an example of how to run the algorithm on one of the input datasets.
alphas = [0.6, 0.4]
betas = [1, 1]
TOPK = 30
CUTOFFN = 25
CUTOFFD = 100

fname = r"football\week9.csv"
rankings, attributes_map, num_attributes, attribute_count = Get_Football(fname)
rankings = rankings[:CUTOFFN]

rankings = [[0, 1, 2, 3, 4, 5], [2, 0, 1, 3, 5, 4], [1, 0, 5, 4, 2, 3]]
attributes_map = {0: 0, 1:0, 2:0, 3:1, 4: 1, 5: 1}

print("Optimal Fair:")
FairILP(alphas, betas, rankings, attributes_map, num_attributes)
print("------------------------")
print("Our algorithm + ILP:")
OurAlgo(alphas, betas, rankings, attributes_map, num_attributes)
print("------------------------")

print("Our algorithm + KS:")

#This code is to set seed for kwiksort, to allow reproducibility
seed_val = 1
for alpha in alphas:
    seed_val *= alpha * 100
rng = np.random.default_rng([9, TOPK, CUTOFFN, len(rankings[0]), int(seed_val)])
OurAlgo_KS(alphas, betas, rankings, attributes_map, num_attributes)
print("------------------------")
print("Best From Input algorithm")
BFI_Algo(alphas, betas, rankings, attributes_map, num_attributes)