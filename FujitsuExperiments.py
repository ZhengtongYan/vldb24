#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from math import prod
import math
import itertools
from math import inf
from sympy.utilities.iterables import multiset_permutations
import Scripts.QUBOGenerator as QUBOGenerator
import Scripts.ProblemGenerator as ProblemGenerator
import Scripts.Postprocessing as Postprocessing

import json
import os
import pathlib
import csv
from os import listdir
from os.path import isfile, join
from pathlib import Path

from multiprocessing import Pool
from multiprocessing.pool import ThreadPool

import time

from dadk.QUBOSolverDAv2 import QUBOSolverDAv2
from dadk.QUBOSolverCPU import *


# In[2]:


def save_to_csv(data, path, filename):
    sd = os.path.abspath(path)
    pathlib.Path(sd).mkdir(parents=True, exist_ok=True) 
    
    f = open(path + '/' + filename, 'a', newline='')
    writer = csv.writer(f)
    writer.writerow(data)
    f.close()


def load_data(path, filename):
    datafile = os.path.abspath(path + '/' + filename)
    if os.path.exists(datafile):
        with open(datafile, 'rb') as file:
            return json.load(file)
        
def load_all_results(path):
    if not os.path.isdir(path):
        return []
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    data = []
    for datafile in onlyfiles:
        with open(path + '/' + datafile, 'rb') as file:
            data.append(json.load(file))
    return data

def save_data(data, path, filename):
    datapath = os.path.abspath(path)
    pathlib.Path(datapath).mkdir(parents=True, exist_ok=True) 
    
    datafile = os.path.abspath(path + '/' + filename)
    mode = 'a' if os.path.exists(datafile) else 'w'
    with open(datafile, mode) as file:
        json.dump(data, file)


# In[3]:


def solve_problem(fujitsu_qubo, da_algorithm='annealing', number_runs=100, number_iterations=1000000, test_with_local_solver=False):
    if test_with_local_solver:
        solver = QUBOSolverCPU(number_runs=number_runs)
    else:
        if da_algorithm == 'annealing':
            solver = QUBOSolverDAv2(optimization_method=da_algorithm, timeout=60, number_iterations=number_iterations, number_runs=number_runs, access_profile_file='annealer.prf', use_access_profile=True)
        else:
            solver = QUBOSolverDAv2(optimization_method=da_algorithm, timeout=60, number_iterations=number_iterations, number_replicas=number_runs, access_profile_file='annealer.prf', use_access_profile=True)

    while True:
        try:
            solution_list = solver.minimize(fujitsu_qubo)
            break
        except:
            print("Library error. Repeating request")

    execution_time = solution_list.execution_time.total_seconds()
    anneal_time = solution_list.anneal_time.total_seconds()
    solutions = solution_list.solutions
    return solutions, execution_time, anneal_time

def parse_solutions_for_serialisation(raw_solutions):
    response = []
    for raw_solution in raw_solutions:
        solution = [raw_solution.configuration, float(raw_solution.frequency), float(raw_solution.energy)]
        response.append(solution)
    return response
                                                
def conduct_synthetic_annealing_experiments(query_types, relations, graph_types, problems, approximation_precisions, penalty_scalings, approximation_types, da_algorithms, iterations_list, problem_path_prefix, result_path_prefix, number_runs=100, number_iterations=1000000, samples = range(0, 20)):
    
    for query_type in query_types:    
        for graph_type in graph_types:
            for i in relations:
                for j in problems:
                    problem_path_main = str(i) + 'relations/' + graph_type + '_graph/' + str(j)
                    card, pred, pred_sel = ProblemGenerator.get_join_ordering_problem(problem_path_prefix + '/' + query_type + '_queries/' + problem_path_main, generated_problems=True)

                    for da_algorithm in da_algorithms:
                        for penalty_scaling in penalty_scalings:
                            for l in range(len(approximation_precisions)):
                                (ap, num_decimal_pos, thres) = approximation_precisions[l]
                                for approximation_type in approximation_types:
                                    if approximation_type == 'quadratic':
                                        fujitsu_qubo, penalty_weight = QUBOGenerator.generate_Fujitsu_QUBO_for_left_deep_trees_v3(card, pred, pred_sel, thres[0], num_decimal_pos, penalty_scaling=penalty_scaling)
                                    elif approximation_type == 'legacy':
                                        fujitsu_qubo, penalty_weight = QUBOGenerator.generate_legacy_Fujitsu_QUBO_for_left_deep_trees(card, pred, pred_sel, thres, num_decimal_pos)
                                    for s in samples:
                                        for number_iterations in iterations_list:
                                            result_path_suffix = 'sample_' + str(s)
                                            result_path = result_path_prefix + '/' + query_type + '_queries/' + da_algorithm + '/' + approximation_type + '_approximation/' + str(number_iterations) + '_iterations/' + str(number_runs) + '_shots/' + problem_path_main + "/thres_config_" + str(ap) + '/' + "penalty_scaling_" + str(penalty_scaling) + '/' + result_path_suffix
                                            if os.path.exists(result_path + '/' + 'response.txt'):
                                                continue
                                            solutions, execution_time, anneal_time = solve_problem(fujitsu_qubo, number_iterations=number_iterations, number_runs=number_runs)
                                            response = parse_solutions_for_serialisation(solutions)
                                            save_data([response, execution_time, anneal_time], result_path, "response.txt")
                                            
                                            thres_path = result_path_prefix + '/' + query_type + '_queries/' + da_algorithm + '/' + approximation_type + '_approximation/' + str(number_iterations) + '_iterations/' + str(number_runs) + '_shots/' + problem_path_main + "/thres_config_" + str(ap)
                                            if not os.path.exists(thres_path + '/thres_config.txt'):
                                                save_data(thres, thres_path, 'thres_config.txt')
                                                                                      
def conduct_benchmark_annealing_experiments(query_types, approximation_precisions, penalty_scalings, approximation_types, da_algorithms, iterations_list, problem_path_prefix, result_path_prefix, number_runs=100, number_iterations=1000000, samples = range(0, 20)):
    
    for query_type in query_types:    
        queries = os.listdir(path=problem_path_prefix + '/' + query_type + '_queries')
        for query in queries:
            problem_path_main = query
            card, pred, pred_sel = ProblemGenerator.get_join_ordering_problem(problem_path_prefix + '/' + query_type + '_queries/' + problem_path_main, generated_problems=True)
            # TODO: Some benchmark queries contain predicate selectivities of 0. These are currently unhandled
            if 0.0 in pred_sel:
                continue
            for da_algorithm in da_algorithms:
                for penalty_scaling in penalty_scalings:
                    for l in range(len(approximation_precisions)):
                        (ap, num_decimal_pos, thres) = approximation_precisions[l]
                        for approximation_type in approximation_types:
                            if approximation_type == 'quadratic':
                                fujitsu_qubo, penalty_weight = QUBOGenerator.generate_Fujitsu_QUBO_for_left_deep_trees_v3(card, pred, pred_sel, thres[0], num_decimal_pos, penalty_scaling=penalty_scaling)
                            elif approximation_type == 'legacy':
                                fujitsu_qubo, penalty_weight = QUBOGenerator.generate_legacy_Fujitsu_QUBO_for_left_deep_trees(card, pred, pred_sel, thres, num_decimal_pos)
                            else:
                                continue
                            for s in samples:
                                for number_iterations in iterations_list:
                                    result_path_suffix = 'sample_' + str(s)
                                    result_path = result_path_prefix + '/' + query_type + '_queries/' + da_algorithm + '/' + approximation_type + '_approximation/' + str(number_iterations) + '_iterations/' + str(number_runs) + '_shots/' + problem_path_main + "/thres_config_" + str(ap) + "/penalty_scaling_" + str(penalty_scaling) + '/' + result_path_suffix
                                    if os.path.exists(result_path + '/' + "response.txt"):
                                        continue
                                    solutions, execution_time, anneal_time = solve_problem(fujitsu_qubo, number_iterations=number_iterations, number_runs=number_runs)
                                    response = parse_solutions_for_serialisation(solutions)
                                    save_data([response, float(execution_time), float(anneal_time)], result_path, "response.txt")                        

                                    thres_path = result_path_prefix + '/' + query_type + '_queries/' + da_algorithm + '/' + approximation_type + '_approximation/' + str(number_iterations) + '_iterations/' + str(number_runs) + '_shots/' + problem_path_main + "/thres_config_" + str(ap)
                                    if not os.path.exists(thres_path + '/thres_config.txt'):
                                        save_data(thres, thres_path, 'thres_config.txt')

def export_synthetic_annealing_results(query_types, relations, graph_types, problems, algorithms, da_algorithms, approximation_types, milp_step_sizes, approximation_precisions, penalty_scalings, iterations_list, considered_thres_configs, problem_path_prefix, benchmark_prefix, milp_prefix, fujitsu_path_prefix, result_path, number_runs=100, samples = range(0, 20), na_cost=20, include_header=True, include_benchmarks=True, include_milp=True, include_annealing=True, include_raw_annealing=True, include_random=True):
    if include_header:
        csv_data = ['method', 'postprocessed', 'milp_step_size', 'query_type', 'num_relations', 'graph_type', 'problem', 'baseline_cost', 'annealing_threshold', 'penalty_scaling', 'num_iterations', 'optimisation_time_in_ms', 'access_time_in_ms', 'cost', 'normalised_cost']
        save_to_csv(csv_data, result_path, 'synthetic_results.txt')     
    
    start = time.time()
    best_costs = inf
    for query_type in query_types:
        for graph_type in graph_types:
            for i in relations:
                for j in problems:
                    csv_data_list = []
                    baseline_cost = inf

                    problem_path_main = str(i) + 'relations/' + graph_type + '_graph/' + str(j)
                    card, pred, pred_sel = ProblemGenerator.get_join_ordering_problem(problem_path_prefix + '/' + query_type + '_queries/' + problem_path_main, generated_problems=True)

                    # Process Benchmark results
                    if include_benchmarks:
                        for (algorithm, tree_type) in algorithms.items():
                            jo_result = load_data(benchmark_prefix + '/' + query_type + '_queries/' + problem_path_main, algorithm + '.json')
                            if jo_result is None:
                                csv_data = [algorithm, 'n/a', 'n/a', query_type, i, graph_type, j, 0, 'n/a', 'n/a', 'n/a', 'n/a', 'n/a', 'n/a', 0]
                                csv_data_list.append(csv_data)
                                continue
                            join_order = jo_result[0]
                            solution_time = jo_result[1]
                            if len(join_order) > 0:
                                if tree_type == 'bushy':
                                    costs = Postprocessing.get_costs_for_bushy_tree(join_order, card, pred, pred_sel)
                                else:
                                    costs = Postprocessing.get_costs_for_leftdeep_tree(join_order, card, pred, pred_sel, {})
                                if costs < baseline_cost:
                                    baseline_cost = costs
                                csv_data = [algorithm, 'n/a', 'n/a', query_type, i, graph_type, j, 0, 'n/a', 'n/a', 'n/a', solution_time, 'n/a', int(costs), 0]
                            else:
                                csv_data = [algorithm, 'n/a', 'n/a', query_type, i, graph_type, j, 0, 'n/a', 'n/a', 'n/a', 'n/a', 'n/a', 'n/a', 0]
                            csv_data_list.append(csv_data)
                            
                    # Process MILP results
                    min_milp_cost = inf
                    best_milp_result = None
                    if include_milp:
                        for step_size in milp_step_sizes:
                            result = load_data(milp_prefix + '/' + query_type + '_queries/' + problem_path_main + '/' + str(step_size) + '_steps/60.0_timeout', 'order.json')
                            if result is None:
                                continue
                            join_order = result[0]
                            solution_time = result[1]
                            if len(join_order) > 0:
                                costs = Postprocessing.get_costs_for_leftdeep_tree(join_order, card, pred, pred_sel, {})
                                if costs < min_milp_cost:
                                    min_milp_cost = costs
                                    best_milp_result = ['milp', 'n/a', step_size, query_type, i, graph_type, j, 0, 'n/a', 'n/a', 'n/a', solution_time, 'n/a', int(costs), 0]
                                if costs < baseline_cost:
                                    baseline_cost = costs
                    
                    if best_milp_result is not None:
                        csv_data_list.append(best_milp_result)
                    else:
                        best_milp_result = ['milp', 'n/a', 'n/a', query_type, i, graph_type, j, 0, 'n/a', 'n/a', 'n/a', solution_time, 'n/a', 'n/a', 0]
                        csv_data_list.append(best_milp_result)
                            
                    # Process Fujitsu results
                    min_annealing_cost = inf
                    best_annealing_result = None
                    if include_annealing:
                        card_dict = {}
                        for approximation_type in approximation_types:
                            min_annealing_cost = inf
                            best_annealing_result = None
                            for da_algorithm in da_algorithms:
                                for penalty_scaling in penalty_scalings:
                                    for number_iterations in iterations_list:
                                        thres_config_path = fujitsu_path_prefix + '/steinbrunn_queries/' + da_algorithm + '/' + approximation_type + '_approximation/' + str(number_iterations) + '_iterations/' + str(number_runs) + '_shots/' + problem_path_main
                                        if not os.path.exists(thres_config_path):
                                            continue
                                        thres_configs = os.listdir(path=thres_config_path)
                                        for thres_config in thres_configs:
                                            if considered_thres_configs is not None and thres_config not in considered_thres_configs:
                                                continue
                                            annealing_thresholds = load_data(thres_config_path + '/' + thres_config, 'thres_config.txt')
                                            if len(annealing_thresholds) > 0:
                                                annealing_threshold = annealing_thresholds[0]
                                            else:
                                                annealing_threshold = 0
                                            for s in samples:
                                                result_path_suffix = 'sample_' + str(s)
                                                fujitsu_result_path = fujitsu_path_prefix + '/' + query_type + '_queries/' + da_algorithm + '/' + approximation_type + '_approximation/' + str(number_iterations) + '_iterations/' + str(number_runs) + '_shots/' + problem_path_main + "/" + thres_config + "/penalty_scaling_" + str(penalty_scaling) + '/' + result_path_suffix
                                                response = load_data(fujitsu_result_path, "response.txt")
                                                if response is None:
                                                    continue
                                                access_time = response[1] * 1000
                                                solution_time = response[2] * 1000
                                                best_solutions_for_time, solutions = Postprocessing.readout(response, card, pred, pred_sel, card_dict)
                                                final_solution = best_solutions_for_time[len(best_solutions_for_time)-1]
                                                annealing_cost = final_solution[1]
                                                if annealing_cost >= min_annealing_cost:
                                                    continue
                                                min_annealing_cost = annealing_cost
                                                postprocessed = final_solution[3]
                                                best_annealing_result = [da_algorithm + '_' + approximation_type, postprocessed, 'n/a', query_type, i, graph_type, j, 0, annealing_threshold, penalty_scaling, number_iterations, solution_time, access_time, annealing_cost, 0]
                                                if annealing_cost < baseline_cost:
                                                    baseline_cost = annealing_cost

                            if best_annealing_result is not None:
                                csv_data_list.append(best_annealing_result)
                    
                    # Export csv data
                    for csv_data in csv_data_list:
                        csv_data[7] = int(baseline_cost)
                        if csv_data[len(csv_data)-2] != 'n/a':
                            normalised_cost = csv_data[len(csv_data)-2]/int(baseline_cost)
                            if normalised_cost > na_cost:
                                csv_data[len(csv_data)-1] = na_cost
                            else:
                                csv_data[len(csv_data)-1] = csv_data[len(csv_data)-2]/int(baseline_cost)
                        else:
                            csv_data[len(csv_data)-1] = na_cost
                        save_to_csv(csv_data, result_path, 'synthetic_results.txt')

def export_synthetic_annealing_times(query_types, relations, graph_types, problems, algorithms, da_algorithms, approximation_types, milp_step_sizes, approximation_precisions, penalty_scalings, iterations_list, considered_thres_configs, problem_path_prefix, benchmark_prefix, milp_prefix, fujitsu_path_prefix, result_path, number_runs=100, samples = range(0, 20), timeout_in_ms=60000, na_cost=20, include_header=True, include_benchmarks=True, include_milp=True, include_annealing=True, include_raw_annealing=True, include_random=True):
    if include_header:
        csv_data = ['method', 'postprocessed', 'milp_step_size', 'query_type', 'num_relations', 'graph_type', 'problem', 'baseline_cost', 'thres_config', 'penalty_scaling', 'num_iterations', 'optimisation_time_in_ms', 'cost', 'normalised_cost']
        save_to_csv(csv_data, result_path, 'synthetic_times.txt')     
    
    start = time.time()
    best_costs = inf
    for query_type in query_types:
        for graph_type in graph_types:
            for i in relations:
                for j in problems:
                    csv_data_list = []
                    baseline_cost = inf

                    problem_path_main = str(i) + 'relations/' + graph_type + '_graph/' + str(j)
                    card, pred, pred_sel = ProblemGenerator.get_join_ordering_problem(problem_path_prefix + '/' + query_type + '_queries/' + problem_path_main, generated_problems=True)

                    # Process Benchmark results
                    if include_benchmarks:
                        for (algorithm, tree_type) in algorithms.items():
                            jo_result = load_data(benchmark_prefix + '/' + query_type + '_queries/' + problem_path_main, algorithm + '.json')
                            if jo_result is None:
                                csv_data = [algorithm, 'n/a', 'n/a', query_type, i, graph_type, j, 0, 'n/a', 'n/a', 'n/a', 0, 'n/a', 0]
                                csv_data_list.append(csv_data)
                                csv_data = [algorithm, 'n/a', 'n/a', query_type, i, graph_type, j, 0, 'n/a', 'n/a', 'n/a', timeout_in_ms, 'n/a', 0]
                                csv_data_list.append(csv_data)
                                continue
                            join_order = jo_result[0]
                            solution_time = jo_result[1]
                            if solution_time > timeout_in_ms:
                                csv_data = [algorithm, 'n/a', 'n/a', query_type, i, graph_type, j, 0, 'n/a', 'n/a', 'n/a', 0, 'n/a', 0]
                                csv_data_list.append(csv_data)
                                csv_data = [algorithm, 'n/a', 'n/a', query_type, i, graph_type, j, 0, 'n/a', 'n/a', 'n/a', timeout_in_ms, 'n/a', 0]
                                csv_data_list.append(csv_data)
                                continue
                            if solution_time != 0:
                                csv_data = [algorithm, 'n/a', 'n/a', query_type, i, graph_type, j, 0, 'n/a', 'n/a', 'n/a', 0, 'n/a', na_cost]
                                csv_data_list.append(csv_data)
                            if len(join_order) > 0:
                                if tree_type == 'bushy':
                                    costs = Postprocessing.get_costs_for_bushy_tree(join_order, card, pred, pred_sel)
                                else:
                                    costs = Postprocessing.get_costs_for_leftdeep_tree(join_order, card, pred, pred_sel, {})
                                if costs < baseline_cost:
                                    baseline_cost = costs
                                csv_data = [algorithm, 'n/a', 'n/a', query_type, i, graph_type, j, 0, 'n/a', 'n/a', 'n/a', solution_time, int(costs), 0]
                                csv_data_list.append(csv_data)
                                csv_data = [algorithm, 'n/a', 'n/a', query_type, i, graph_type, j, 0, 'n/a', 'n/a', 'n/a', timeout_in_ms, int(costs), 0]
                                csv_data_list.append(csv_data)
                            else:
                                csv_data = [algorithm, 'n/a', 'n/a', query_type, i, graph_type, j, 0, 'n/a', 'n/a', 'n/a', 0, 'n/a', 0]
                                csv_data_list.append(csv_data)
                                csv_data = [algorithm, 'n/a', 'n/a', query_type, i, graph_type, j, 0, 'n/a', 'n/a', 'n/a', timeout_in_ms, 'n/a', 0]
                                csv_data_list.append(csv_data)
                            
                    # Process MILP results
                    if include_milp:
                        milp_step_results = {}
                        for step_size in milp_step_sizes:
                            milp_step_results[step_size] = []
                            milp_path = milp_prefix + '/' + query_type + '_queries/' + problem_path_main + '/' + str(step_size) + '_steps'
                            if not os.path.exists(milp_path):
                                continue
                            timeout_strings = os.listdir(path=milp_path)
                            timeouts = []
                            for timeout_string in timeout_strings:
                                timeout = float(timeout_string.split('_')[0])
                                timeouts.append(timeout)
                            timeouts = list(sorted(timeouts))
                            if timeouts[0] != 0:
                                milp_result = ['milp', 'n/a', step_size, query_type, i, graph_type, j, 0, 'n/a', 'n/a', 'n/a', 0, 'n/a', na_cost]
                                milp_step_results[step_size].append(milp_result)

                            min_milp_cost = inf
                            min_milp_time = None
                            min_milp_result = None
                            for timeout in timeouts:
                                result = load_data(milp_prefix + '/' + query_type + '_queries/' + problem_path_main + '/' + str(step_size) + '_steps/' + str(timeout) + '_timeout', 'order.json')
                                if result is None or len(result[0]) == 0:
                                    min_milp_result = ['milp', 'n/a', step_size, query_type, i, graph_type, j, 0, 'n/a', 'n/a', 'n/a', timeout*1000, 'n/a', 0]
                                    milp_step_results[step_size].append(min_milp_result)
                                    continue
                                join_order = result[0]
                                solution_time = result[1]
                                if solution_time > timeout_in_ms:
                                    # The MILP solver often terminates only a few ms after the timeout
                                    # Hence, we allow up to one second time overhead
                                    if solution_time < timeout_in_ms + 1000:
                                        solution_time = timeout_in_ms
                                    else:
                                        continue
                                if len(join_order) > 0:
                                    costs = Postprocessing.get_costs_for_leftdeep_tree(join_order, card, pred, pred_sel, {})
                                    if costs < min_milp_cost:
                                        if solution_time > timeout:
                                            solution_time = timeout*1000
                                        min_milp_time = solution_time
                                        min_milp_result = ['milp', 'n/a', step_size, query_type, i, graph_type, j, 0, 'n/a', 'n/a', 'n/a', solution_time, int(costs), 0]
                                        min_milp_cost = costs
                                        milp_step_results[step_size].append(min_milp_result)
                                    elif costs == min_milp_cost:
                                        min_milp_result = ['milp', 'n/a', step_size, query_type, i, graph_type, j, 0, 'n/a', 'n/a', 'n/a', timeout*1000, int(costs), 0]
                                        milp_step_results[step_size].append(min_milp_result)
                                    if costs < baseline_cost:
                                        baseline_cost = costs
                            if min_milp_result is not None:
                                min_milp_result = min_milp_result.copy()
                                min_milp_result[11] = timeout_in_ms
                                milp_step_results[step_size].append(min_milp_result)
                            else:
                                milp_result = ['milp', 'n/a', step_size, query_type, i, graph_type, j, 0, 'n/a', 'n/a', 'n/a', timeout_in_ms, 'n/a', na_cost]
                                milp_step_results[step_size].append(milp_result)
                        
                        # Export the best MILP results obtained from all step sizes
                        # For equal cost, we prefer higher step sizes, which tend to beget lower optimisation times
                        best_step_size = max(milp_step_sizes)
                        min_step_cost = inf
                        for (step_size, milp_step_result) in milp_step_results.items():
                            final_result = milp_step_result[len(milp_step_result)-1]
                            final_result_cost = final_result[len(final_result)-2]
                            if final_result_cost == 'n/a':
                                continue
                            if final_result_cost < min_step_cost:
                                min_step_cost = final_result_cost
                                best_step_size = step_size
                            elif final_result_cost == min_step_cost and step_size > best_step_size:
                                best_step_size = step_size
                        best_milp_results = milp_step_results[best_step_size]
                        for best_milp_result in best_milp_results:
                            csv_data_list.append(best_milp_result)
                            
                    # Process Fujitsu results
                    if include_annealing:
                        card_dict = {}
                        for approximation_type in approximation_types:
                            for da_algorithm in da_algorithms:
                                for penalty_scaling in penalty_scalings:
                                    for number_iterations in iterations_list:
                                        thres_config_path = fujitsu_path_prefix + '/steinbrunn_queries/' + da_algorithm + '/' + approximation_type + '_approximation/' + str(number_iterations) + '_iterations/' + str(number_runs) + '_shots/' + problem_path_main
                                        if not os.path.exists(thres_config_path):
                                            continue
                                        thres_configs = os.listdir(path=thres_config_path)
                                        annealing_thres_results = {}
                                        min_thres_cost = inf
                                        best_thres_config = None
                                        for thres_config in thres_configs:
                                            if considered_thres_configs is not None and thres_config not in considered_thres_configs:
                                                continue
                                            if best_thres_config is None:
                                                best_thres_config = thres_config
                                            annealing_thres_results[thres_config] = []
                                            annealing_result = [da_algorithm + '_' + approximation_type, 1, 'n/a', query_type, i, graph_type, j, 0, thres_config, penalty_scaling, number_iterations, 0, 'n/a', na_cost]
                                            annealing_thres_results[thres_config].append(annealing_result)

                                            solution_time = 0
                                            min_annealing_cost = inf
                                            min_annealing_result = None
                                            for s in samples:
                                                result_path_suffix = 'sample_' + str(s)
                                                fujitsu_result_path = fujitsu_path_prefix + '/' + query_type + '_queries/' + da_algorithm + '/' + approximation_type + '_approximation/' + str(number_iterations) + '_iterations/' + str(number_runs) + '_shots/' + problem_path_main + "/" + thres_config + "/penalty_scaling_" + str(penalty_scaling) + '/' + result_path_suffix
                                                response = load_data(fujitsu_result_path, "response.txt")
                                                if response is None:
                                                    continue
                                                access_time = response[1] * 1000
                                                solution_time = solution_time + access_time
                                                best_solutions_for_time, solutions = Postprocessing.readout(response, card, pred, pred_sel, card_dict)
                                                final_solution = best_solutions_for_time[len(best_solutions_for_time)-1]
                                                    
                                                annealing_cost = final_solution[1]
                                                if annealing_cost >= min_annealing_cost:
                                                    continue
                                                min_annealing_cost = annealing_cost
                                                readout_time = final_solution[2]
                                                postprocessed = final_solution[3]
                                                if (solution_time + readout_time) > timeout_in_ms:
                                                    continue
                                                annealing_result = [da_algorithm + '_' + approximation_type, postprocessed, 'n/a', query_type, i, graph_type, j, 0, thres_config, penalty_scaling, number_iterations, solution_time + readout_time, annealing_cost, 0]
                                                annealing_thres_results[thres_config].append(annealing_result)  
                                                min_annealing_result = annealing_result
                                                if annealing_cost < min_thres_cost:
                                                    min_thres_cost = annealing_cost
                                                    best_thres_config = thres_config
                                                if annealing_cost < baseline_cost:
                                                    baseline_cost = annealing_cost
                                            
                                            min_annealing_result = min_annealing_result.copy()
                                            min_annealing_result[len(min_annealing_result)-3] = timeout_in_ms
                                            annealing_thres_results[thres_config].append(min_annealing_result)
                                        best_annealing_results = annealing_thres_results[best_thres_config]
                                        for best_annealing_result in best_annealing_results:
                                            csv_data_list.append(best_annealing_result)

                    # Export csv data
                    for csv_data in csv_data_list:
                        csv_data[7] = int(baseline_cost)
                        if csv_data[len(csv_data)-2] != 'n/a':
                            normalised_cost = csv_data[len(csv_data)-2]/int(baseline_cost)
                            if normalised_cost > na_cost:
                                csv_data[len(csv_data)-1] = na_cost
                            else:
                                csv_data[len(csv_data)-1] = csv_data[len(csv_data)-2]/int(baseline_cost)
                        else:
                            csv_data[len(csv_data)-1] = na_cost
                        save_to_csv(csv_data, result_path, 'synthetic_times.txt')     
                
def export_benchmark_annealing_results(query_types, algorithms, da_algorithms, approximation_types, milp_step_sizes, penalty_scalings, iterations_list, considered_thres_configs, problem_path_prefix, benchmark_prefix, milp_prefix, fujitsu_path_prefix, result_path, number_runs=100, samples = range(0, 20), na_cost=20, include_header=True, include_benchmarks=True, include_milp=True, include_annealing=True, include_raw_annealing=True, include_random=True):
    if include_header:
        csv_data = ['method', 'postprocessed', 'milp_step_size', 'query_type', 'query', 'baseline_cost', 'penalty_scaling', 'num_iterations', 'optimisation_time_in_ms', 'access_time_in_ms', 'cost', 'normalised_cost']
        save_to_csv(csv_data, result_path, 'benchmark_results.txt')     
    
    start = time.time()
    best_costs = inf
    for query_type in query_types:
        queries = os.listdir(path=problem_path_prefix + '/' + query_type + '_queries')
        for query in queries:
            query_number = int(query.split('q')[1])
            problem_path_main = query
        
            csv_data_list = []
            baseline_cost = inf

            card, pred, pred_sel = ProblemGenerator.get_join_ordering_problem(problem_path_prefix + '/' + query_type + '_queries/' + problem_path_main, generated_problems=True)
      
            # Process Benchmark results
            if include_benchmarks:
                for (algorithm, tree_type) in algorithms.items():
                    jo_result = load_data(benchmark_prefix + '/' + query_type + '_queries/' + problem_path_main, algorithm + '.json')
                    if jo_result is None:
                        continue
                    join_order = jo_result[0]
                    solution_time = jo_result[1]
                    if len(join_order) > 0:
                        if tree_type == 'bushy':
                            costs = Postprocessing.get_costs_for_bushy_tree(join_order, card, pred, pred_sel)
                        else:
                            costs = Postprocessing.get_costs_for_leftdeep_tree(join_order, card, pred, pred_sel, {})
                        #print(costs)
                        if costs < baseline_cost:
                            baseline_cost = costs
                        csv_data = [algorithm, 'n/a', 'n/a', query_type, query_number, 0, 'n/a', 'n/a', solution_time, 'n/a', int(costs), 0]
                    else:
                        csv_data = [algorithm, 'n/a', 'n/a', query_type, query_number, 0, 'n/a', 'n/a', 'n/a', 'n/a', 'n/a', 0]
                    csv_data_list.append(csv_data)
                            
            # Process MILP results
            min_milp_cost = inf
            best_milp_result = None
            if include_milp:
                for step_size in milp_step_sizes:
                    result = load_data(milp_prefix + '/' + query_type + '_queries/' + problem_path_main + '/' + str(step_size) + '_steps', 'order.json')
                    if result is None:
                        continue
                    join_order = result[0]
                    solution_time = result[1]
                    if len(join_order) > 0:
                        costs = Postprocessing.get_costs_for_leftdeep_tree(join_order, card, pred, pred_sel, {})
                        if costs < min_milp_cost:
                            min_milp_cost = costs
                            best_milp_result = ['milp', 'n/a', step_size, query_type, query_number, 0, 'n/a', 'n/a', solution_time, 'n/a', int(costs), 0]
                        if costs < baseline_cost:
                            baseline_cost = costs
            
            if best_milp_result is not None:
                csv_data_list.append(best_milp_result)
                                    
            # Process Fujitsu results
            if include_annealing:
                card_dict = {}
                for approximation_type in approximation_types:
                    min_annealing_cost = inf
                    best_annealing_result = None
                    for da_algorithm in da_algorithms:
                        for penalty_scaling in penalty_scalings:
                            for number_iterations in iterations_list:
                                thres_config_path = fujitsu_path_prefix + '/' + query_type + '_queries/' + da_algorithm + '/' + approximation_type + '_approximation/' + str(number_iterations) + '_iterations/' + str(number_runs) + '_shots/' + problem_path_main
                                if not os.path.exists(thres_config_path):
                                    continue
                                thres_configs = os.listdir(path=thres_config_path)
                                for thres_config in thres_configs:
                                    if considered_thres_configs is not None and thres_config not in considered_thres_configs:
                                        continue
                                    for s in samples:
                                        result_path_suffix = 'sample_' + str(s)
                                        fujitsu_result_path = fujitsu_path_prefix + '/' + query_type + '_queries/' + da_algorithm + '/' + approximation_type + '_approximation/' + str(number_iterations) + '_iterations/' + str(number_runs) + '_shots/' + problem_path_main + "/" + thres_config + "/penalty_scaling_" + str(penalty_scaling) + '/' + result_path_suffix
                                        response = load_data(fujitsu_result_path, "response.txt")
                                        if response is None:
                                            continue
                                        access_time = response[1] * 1000
                                        solution_time = response[2] * 1000
                                        
                                        best_solutions_for_time, solutions = Postprocessing.readout(response, card, pred, pred_sel, card_dict)
                                        final_solution = best_solutions_for_time[len(best_solutions_for_time)-1]
                                        annealing_cost = final_solution[1]
                                        if annealing_cost >= min_annealing_cost:
                                            continue
                                        min_annealing_cost = annealing_cost
                                        postprocessed = final_solution[3]
                                        best_annealing_result = [da_algorithm + '_' + approximation_type, postprocessed, 'n/a', query_type, query_number, 0, penalty_scaling, number_iterations, solution_time, access_time, annealing_cost, 0]                                        
                                        if annealing_cost < baseline_cost:
                                            baseline_cost = annealing_cost
                                        
                    if best_annealing_result is not None:
                        csv_data_list.append(best_annealing_result)     

            # Export csv data
            for csv_data in csv_data_list:
                csv_data[5] = int(baseline_cost)
                if csv_data[len(csv_data)-2] != 'n/a':
                    normalised_cost = csv_data[len(csv_data)-2]/int(baseline_cost)
                    if normalised_cost > na_cost:
                        csv_data[len(csv_data)-1] = na_cost
                    else:
                        csv_data[len(csv_data)-1] = csv_data[len(csv_data)-2]/int(baseline_cost)
                else:
                    csv_data[len(csv_data)-1] = na_cost
                save_to_csv(csv_data, result_path, 'benchmark_results.txt')


# In[4]:


def conduct_experiment():
    
    relations = [18, 22, 26, 30, 34, 38, 42, 46, 50]
    
    graph_types = ['CHAIN', 'STAR', 'CYCLE']
    
    problems = range(10)

    penalty_scalings = [2]
            
    iterations_list = [1000000]
    
    problem_path_prefix = 'ExperimentalAnalysis/Fujitsu/Problems'
    fujitsu_path_prefix = 'ExperimentalAnalysis/Fujitsu/Results'
    benchmark_prefix = 'ExperimentalAnalysis/Benchmarks/Results'
    milp_prefix = 'ExperimentalAnalysis/MILP/Results'
    result_path = 'ExperimentalAnalysis'
    algorithms = {'ikkbz': 'leftdeep', 'dpsizelinear': 'leftdeep'}
    
    da_algorithms = ['annealing']
    
    milp_step_sizes = [2, 10, 100]
    
    thres_configs = ['thres_config_4', 'thres_config_5', 'thres_config_6', 'thres_config_12']
    
    postprocessing_methods = [1, 2]
    
    number_runs = 100
    samples = range(10)
    
    include_header = True
    include_benchmarks = True
    include_milp = True
    include_annealing = True
    include_raw_annealing = True
    include_random = False
    aggregate_annealing_results = True
    
    approximation_types = ['quadratic']
    
    ## Synthetic experiments
    
    query_types = ['steinbrunn']
   
    # Approximation precisions for our novel QUBO encoding, formated as (config_id, #decimal_positions, thresholds)
    approximation_precisions = [(4, 2, [0.63]), (5, 2, [2.55]), (6, 2, [5.11]), (12, 0, [])]
    conduct_synthetic_annealing_experiments(query_types, relations, graph_types, problems, approximation_precisions, penalty_scalings, approximation_types, da_algorithms, iterations_list, problem_path_prefix, fujitsu_path_prefix, number_runs=number_runs, samples = samples)
    export_synthetic_annealing_results(query_types, relations, graph_types, problems, algorithms, da_algorithms, approximation_types, milp_step_sizes, approximation_precisions, penalty_scalings, iterations_list, thres_configs, problem_path_prefix, benchmark_prefix, milp_prefix, fujitsu_path_prefix, result_path, number_runs=number_runs, samples = samples, include_header=include_header, include_benchmarks=include_benchmarks, include_milp=include_milp, include_annealing=include_annealing, include_raw_annealing=include_raw_annealing, include_random=include_random)
    export_synthetic_annealing_times(query_types, relations, graph_types, problems, algorithms, da_algorithms, approximation_types, milp_step_sizes, approximation_precisions, penalty_scalings, iterations_list, thres_configs, problem_path_prefix, benchmark_prefix, milp_prefix, fujitsu_path_prefix, result_path, number_runs=number_runs, samples = samples, include_header=include_header, include_benchmarks=include_benchmarks, include_milp=include_milp, include_annealing=include_annealing, include_raw_annealing=include_raw_annealing, include_random=include_random)
    
    ## Benchmark experiments
    
    query_types = ['sqlite']
    
    # Approximation precisions for our novel QUBO encoding, formated as (config_id, #decimal_positions, thresholds)
    approximation_precisions = [(12, 0, [])]
    conduct_benchmark_annealing_experiments(query_types, approximation_precisions, penalty_scalings, approximation_types, da_algorithms, iterations_list, problem_path_prefix, fujitsu_path_prefix, number_runs=number_runs, samples = samples)
    export_benchmark_annealing_results(query_types, algorithms, da_algorithms, approximation_types, milp_step_sizes, penalty_scalings, iterations_list, thres_configs, problem_path_prefix, benchmark_prefix, milp_prefix, fujitsu_path_prefix, result_path, number_runs=number_runs, samples = samples, include_header=include_header, include_benchmarks=include_benchmarks, include_milp=include_milp, include_annealing=include_annealing, include_raw_annealing=include_raw_annealing, include_random=include_random)

if __name__ == "__main__":
    conduct_experiment()

