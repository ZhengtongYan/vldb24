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
import Scripts.DataExport as DataExport

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

import neal


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


# In[1]:


def solve_problem(qubo, number_runs=100, number_iterations=1000):
    
    sampler = neal.SimulatedAnnealingSampler()
    start = time.time()
    result = sampler.sample(qubo, num_reads=number_runs, num_sweeps=number_iterations, answer_mode='raw')
    opt_time = time.time() - start
    opt_time_in_ms = opt_time * 1000
    
    solutions = []
    for item in result.record:
        bitstring = [int(x) for x in item[0]]
        solutions.append([bitstring, int(item[2]), float(item[1])])
        
    return solutions, opt_time_in_ms

def get_combined_cost_evolution(annealing_thres_results, best_thres_config):
    combined_results = []
    for (k, v) in annealing_thres_results.items():
        if list(k)[1] == best_thres_config:
            combined_results.extend(v)
    combined_results = sorted(combined_results, key=lambda x: x['time'])
    min_costs = inf
    filtered_result = []
    while len(combined_results) != 0:
        combined_result = combined_results.pop(0)
        if combined_result["costs"] < min_costs:
            min_costs = combined_result["costs"]
            filtered_result.append(combined_result)
    
    return filtered_result

def conduct_benchmark_experiments(approximation_types, penalty_scalings, iterations_list, approximation_precisions, problem_path_prefix, result_path_prefix, number_runs=100, max_num_samples=20, timeout_in_ms=60000):
    benchmarks = os.listdir(path=problem_path_prefix + '/benchmark_queries')
    for benchmark in benchmarks:
        queries = os.listdir(path=problem_path_prefix + '/benchmark_queries/' + benchmark)
        for query in queries:
            query_number = int(query.split('q')[1])
            card, pred, pred_sel = ProblemGenerator.get_join_ordering_problem(problem_path_prefix + '/benchmark_queries/' + benchmark + '/' + query, generated_problems=True)
            if 0.0 in pred_sel:
                continue
            for penalty_scaling in penalty_scalings:
                for l in range(len(approximation_precisions)):
                    (ap, num_decimal_pos, thres) = approximation_precisions[l]
                    for approximation_type in approximation_types:
                        if approximation_type == 'quadratic':
                            if len(thres) == 0:
                                qubo, penalty_weight = QUBOGenerator.generate_DWave_QUBO_for_left_deep_trees_v2(card, pred, pred_sel, penalty_scaling=penalty_scaling)
                            else:
                                qubo, penalty_weight = QUBOGenerator.generate_DWave_QUBO_for_left_deep_trees(card, pred, pred_sel, thres[0], num_decimal_pos, penalty_scaling=penalty_scaling)
                        elif approximation_type == 'legacy':
                            qubo, penalty_weight = QUBOGenerator.generate_DWave_legacy_QUBO_for_left_deep_trees(card, pred, pred_sel, thres, num_decimal_pos, penalty_scaling=penalty_scaling)
                        
                        for number_iterations in iterations_list:
                            total_annealing_time = 0
                            sample_index = 0
                            while sample_index < max_num_samples and total_annealing_time < timeout_in_ms:
                                result_path = result_path_prefix + '/simulated_annealing/benchmark_queries/' + benchmark + '/' + query + '/' + approximation_type + '_approximation/' + '/thres_config_' + str(ap) + '/penalty_scaling_' + str(penalty_scaling) + '/' + str(number_iterations) + '_iterations/' + str(number_runs) + '_shots/' + 'sample_' + str(sample_index)
                                solutions, opt_time = solve_problem(qubo, number_runs=number_runs, number_iterations=number_iterations)
                                total_annealing_time = total_annealing_time + opt_time
                                save_data([solutions, float(opt_time)], result_path, "response.txt") 
                                thres_path = result_path_prefix + '/simulated_annealing/benchmark_queries/' + benchmark + '/' + query + '/' + approximation_type + '_approximation/' + '/thres_config_' + str(ap)
                                if not os.path.exists(thres_path + '/thres_config.txt'):
                                    save_data(thres, thres_path, 'thres_config.txt')
                                sample_index = sample_index + 1 
                                
def process_benchmark_annealing_results(approximation_types, penalty_scalings, iterations_list, considered_thres_configs, problem_path_prefix, data_path_prefix, result_path_prefix, number_runs=100, max_num_samples=20, replace_existing_results=False, timeout_in_ms = 60000):
    benchmarks = os.listdir(path=problem_path_prefix + '/benchmark_queries')
    for benchmark in benchmarks:
        queries = os.listdir(path=problem_path_prefix + '/benchmark_queries/' + benchmark)
        for query in queries:
            query_number = int(query.split('q')[1])
            card, pred, pred_sel = ProblemGenerator.get_join_ordering_problem(problem_path_prefix + '/benchmark_queries/' + benchmark + '/' + query, generated_problems=True)
            if 0.0 in pred_sel:
                continue
            card_dict = {}
            for approximation_type in approximation_types:
                for penalty_scaling in penalty_scalings:
                    annealing_thres_results = {}
                    min_cost = inf
                    best_config = None
                    for number_iterations in iterations_list:
                        result_path = result_path_prefix + '/benchmark_queries/' + benchmark + '/' + query 
                        result_file = result_path +  '/simulated_annealing_' + approximation_type + '.json'
                        if os.path.exists(result_file):
                            if replace_existing_results:
                                try:
                                    os.remove(result_file)
                                except OSError:
                                    pass
                            else:
                                continue
                                            
                        thres_config_path = data_path_prefix + '/simulated_annealing/benchmark_queries/' + benchmark + '/' + query + '/' + approximation_type + '_approximation'
                        if not os.path.exists(thres_config_path):
                            continue
                        thres_configs = os.listdir(path=thres_config_path)
                        for thres_config in thres_configs:
                            if considered_thres_configs is not None and thres_config not in considered_thres_configs:
                                continue
                            if best_config is None:
                                best_config = frozenset([number_iterations, thres_config])
                                
                            annealing_thres_results[frozenset([number_iterations, thres_config])] = []

                            solution_time = 0
                            min_annealing_cost = inf
                            min_annealing_result = None
                            for s in range(max_num_samples):
                                data_path_suffix = 'sample_' + str(s)
                                data_path = data_path_prefix + '/simulated_annealing/benchmark_queries/' + benchmark + '/' + query + '/' + approximation_type + '_approximation/' + thres_config + '/penalty_scaling_' + str(penalty_scaling) + '/' + str(number_iterations) + '_iterations/' + str(number_runs) + '_shots/' + data_path_suffix 
                                if not os.path.exists(data_path):
                                    continue
                                response = load_data(data_path, "response.txt")
                                if response is None:
                                    continue
                                solution_time = solution_time + response[1]
                                best_solutions_for_time, solutions = Postprocessing.readout(response, card, pred, pred_sel, card_dict)
                                final_solution = best_solutions_for_time[len(best_solutions_for_time)-1]

                                join_order = [int(x) for x in final_solution[0]]
                                annealing_cost = final_solution[1]
                                readout_time = final_solution[2]
                                postprocessed = final_solution[3]
                                total_time = solution_time + readout_time
                                if annealing_cost >= min_annealing_cost or total_time > timeout_in_ms:
                                    continue
                                min_annealing_cost = annealing_cost

                                annealing_thres_results[frozenset([number_iterations, thres_config])].append({"time": total_time, "join_order":join_order, "costs": annealing_cost, "fallback": postprocessed})
                                if annealing_cost < min_cost:
                                    min_cost = annealing_cost
                                    best_config = frozenset([number_iterations, thres_config])

                    #best_annealing_results = annealing_thres_results[best_config]
                    if best_config is None:
                        save_data([], result_path, 'simulated_annealing_' + approximation_type + '.json')
                    else:
                        best_annealing_results = get_combined_cost_evolution(annealing_thres_results, list(best_config)[1])
                        save_data(best_annealing_results, result_path, 'simulated_annealing_' + approximation_type + '.json')
            

def conduct_synthetic_experiments(approximation_types, penalty_scalings, iterations_list, approximation_precisions, problem_path_prefix, result_path_prefix, number_runs=100, max_num_samples=20, timeout_in_ms=60000):
    graph_types = os.listdir(path=problem_path_prefix + '/synthetic_queries/')
    for graph_type_string in graph_types:
        graph_type = graph_type_string.split("_")[0]
        relations = os.listdir(path=problem_path_prefix + '/synthetic_queries/' + graph_type + '_graph')
        for relations_string in relations:
            i = int(relations_string.split("relations")[0])
            problems = os.listdir(path=problem_path_prefix + '/synthetic_queries/' + graph_type + '_graph/' + str(i) + 'relations')
            for j in problems:
                j = int(j)

                problem_path_main = graph_type + '_graph/' + str(i) + 'relations/' + str(j)
                card, pred, pred_sel = ProblemGenerator.get_join_ordering_problem(problem_path_prefix + '/synthetic_queries/' + problem_path_main, generated_problems=True)
                if 0.0 in pred_sel:
                    continue
                for penalty_scaling in penalty_scalings:
                    for l in range(len(approximation_precisions)):
                        (ap, num_decimal_pos, thres) = approximation_precisions[l]
                        for approximation_type in approximation_types:
                            if approximation_type == 'quadratic':
                                if len(thres) == 0:
                                    qubo, penalty_weight = QUBOGenerator.generate_DWave_QUBO_for_left_deep_trees_v2(card, pred, pred_sel, penalty_scaling=penalty_scaling)
                                else:
                                    qubo, penalty_weight = QUBOGenerator.generate_DWave_QUBO_for_left_deep_trees(card, pred, pred_sel, thres[0], num_decimal_pos, penalty_scaling=penalty_scaling)
                            elif approximation_type == 'legacy':
                                qubo, penalty_weight = QUBOGenerator.generate_DWave_legacy_QUBO_for_left_deep_trees(card, pred, pred_sel, thres, num_decimal_pos, penalty_scaling=penalty_scaling)
                            
                            for number_iterations in iterations_list:
                                total_annealing_time = 0
                                sample_index = 0
                                while sample_index < max_num_samples and total_annealing_time < timeout_in_ms:
                                    result_path = result_path_prefix + '/simulated_annealing/synthetic_queries/' + graph_type_string + '/' + relations_string + '/' + str(j) + '/' + approximation_type + '_approximation/' + '/thres_config_' + str(ap) + '/penalty_scaling_' + str(penalty_scaling) + '/' + str(number_iterations) + '_iterations/' + str(number_runs) + '_shots/' + 'sample_' + str(sample_index)
                                    solutions, opt_time = solve_problem(qubo, number_runs=number_runs, number_iterations=number_iterations)
                                    total_annealing_time = total_annealing_time + opt_time
                                    save_data([solutions, float(opt_time)], result_path, "response.txt") 
                                    thres_path = result_path_prefix + '/simulated_annealing/synthetic_queries/' + graph_type_string + '/' + relations_string + '/' + str(j) + '/' + approximation_type + '_approximation/' + '/thres_config_' + str(ap)
                                    if not os.path.exists(thres_path + '/thres_config.txt'):
                                        save_data(thres, thres_path, 'thres_config.txt')
                                    sample_index = sample_index + 1
                                                                     
def process_synthetic_annealing_results(approximation_types, penalty_scalings, iterations_list, considered_thres_configs, problem_path_prefix, data_path_prefix, result_path_prefix, number_runs=100, max_num_samples=20, replace_existing_results=False, timeout_in_ms=60000):
    graph_types = os.listdir(path=problem_path_prefix + '/synthetic_queries/')
    for graph_type_string in graph_types:
        graph_type = graph_type_string.split("_")[0]
        relations = os.listdir(path=problem_path_prefix + '/synthetic_queries/' + graph_type + '_graph')
        for relations_string in relations:
            i = int(relations_string.split("relations")[0])
            problems = os.listdir(path=problem_path_prefix + '/synthetic_queries/' + graph_type + '_graph/' + str(i) + 'relations')
            for j in problems:
                j = int(j)

                problem_path_main = graph_type + '_graph/' + str(i) + 'relations/' + str(j)
                card, pred, pred_sel = ProblemGenerator.get_join_ordering_problem(problem_path_prefix + '/synthetic_queries/' + problem_path_main, generated_problems=True)
                if 0.0 in pred_sel:
                    continue
                card_dict = {}
                for approximation_type in approximation_types:
                    for penalty_scaling in penalty_scalings:
                        annealing_thres_results = {}
                        min_cost = inf
                        best_config = None
                        for number_iterations in iterations_list:
                            result_path = result_path_prefix + '/synthetic_queries/' + graph_type + '_graph/' + str(i) + 'relations/' + str(j)
                            result_file = result_path + '/simulated_annealing_' + approximation_type + '.json'
                            if os.path.exists(result_file):
                                if replace_existing_results:
                                    try:
                                        os.remove(result_file)
                                    except OSError:
                                        pass
                                else:
                                    continue
                                            
                            thres_config_path = data_path_prefix + '/simulated_annealing/synthetic_queries/' + problem_path_main + '/' + approximation_type + '_approximation'
                            if not os.path.exists(thres_config_path):
                                continue
                            thres_configs = os.listdir(path=thres_config_path)
                            for thres_config in thres_configs:
                                if considered_thres_configs is not None and thres_config not in considered_thres_configs:
                                    continue
                                if best_config is None:
                                    best_config = frozenset([number_iterations, thres_config])
                                
                                annealing_thres_results[frozenset([number_iterations, thres_config])] = []

                                solution_time = 0
                                min_annealing_cost = inf
                                min_annealing_result = None
                                for s in range(max_num_samples):
                                    data_path_suffix = 'sample_' + str(s)
                                    data_path = data_path_prefix + '/simulated_annealing/synthetic_queries/' + problem_path_main + '/' + approximation_type + '_approximation/' + thres_config + '/penalty_scaling_' + str(penalty_scaling) + '/' + str(number_iterations) + '_iterations/' + str(number_runs) + '_shots/' + data_path_suffix 
                                    if not os.path.exists(data_path):
                                        continue
                                    response = load_data(data_path, "response.txt")
                                    if response is None:
                                        continue
                                    solution_time = solution_time + response[1]
                                    best_solutions_for_time, solutions = Postprocessing.readout(response, card, pred, pred_sel, card_dict)
                                    final_solution = best_solutions_for_time[len(best_solutions_for_time)-1]

                                    join_order = [int(x) for x in final_solution[0]] 
                                    annealing_cost = final_solution[1]
                                    readout_time = final_solution[2]
                                    postprocessed = final_solution[3]
                                    total_time = solution_time + readout_time
                                    if annealing_cost >= min_annealing_cost or total_time > timeout_in_ms:
                                        continue
                                        
                                    min_annealing_cost = annealing_cost

                                    annealing_thres_results[frozenset([number_iterations, thres_config])].append({"time": total_time, "join_order":join_order, "costs": annealing_cost, "fallback": postprocessed})
                                    if annealing_cost < min_cost:
                                        min_cost = annealing_cost
                                        best_config = frozenset([number_iterations, thres_config])

                        #best_annealing_results = annealing_thres_results[best_config]
                        if best_config is None:
                            save_data([], result_path, 'simulated_annealing_' + approximation_type + '.json')
                        else:
                            best_annealing_results = get_combined_cost_evolution(annealing_thres_results, list(best_config)[1])
                            save_data(best_annealing_results, result_path, 'simulated_annealing_' + approximation_type + '.json')

if __name__ == "__main__":
    
    penalty_scalings = [2]
    iterations_list = [100, 1000, 10000]
    number_runs = 100
    max_num_samples = 10
    timeout_in_ms = 60000
    problem_path_prefix = 'ExperimentalAnalysis/Problems'
    result_path_prefix = 'ExperimentalAnalysis/Data'
    
    approximation_types = ['quadratic']
    approximation_precisions = [(4, 2, [0.63]), (5, 2, [2.55]), (6, 2, [5.11]), (12, 0, [])]
    conduct_synthetic_experiments(approximation_types, penalty_scalings, iterations_list, approximation_precisions, problem_path_prefix, result_path_prefix, number_runs=number_runs, max_num_samples=max_num_samples, timeout_in_ms=timeout_in_ms)
    
    considered_thres_configs = ['thres_config_4', 'thres_config_5', 'thres_config_6', 'thres_config_12']
    data_path_prefix = 'ExperimentalAnalysis/Data'
    result_path_prefix = 'ExperimentalAnalysis/Results'
    replace_existing_results = True
    process_synthetic_annealing_results(approximation_types, penalty_scalings, iterations_list, considered_thres_configs, problem_path_prefix, data_path_prefix, result_path_prefix, number_runs=number_runs, max_num_samples=max_num_samples, replace_existing_results=replace_existing_results, timeout_in_ms=timeout_in_ms)
    
    penalty_scalings = [2]
    iterations_list = [100, 1000, 10000]
    number_runs = 100
    max_num_samples = 10
    timeout_in_ms = 60000
    problem_path_prefix = 'ExperimentalAnalysis/Problems'
    result_path_prefix = 'ExperimentalAnalysis/Data'
    
    approximation_types = ['quadratic']
    approximation_precisions = [(4, 2, [0.63]), (5, 2, [2.55]), (6, 2, [5.11]), (12, 0, [])]
    conduct_benchmark_experiments(approximation_types, penalty_scalings, iterations_list, approximation_precisions, problem_path_prefix, result_path_prefix, number_runs=number_runs, max_num_samples=max_num_samples, timeout_in_ms=timeout_in_ms)
    
    considered_thres_configs = ['thres_config_4', 'thres_config_5', 'thres_config_6', 'thres_config_12']
    data_path_prefix = 'ExperimentalAnalysis/Data'
    result_path_prefix = 'ExperimentalAnalysis/Results'
    replace_existing_results = True
    process_benchmark_annealing_results(approximation_types, penalty_scalings, iterations_list, considered_thres_configs, problem_path_prefix, data_path_prefix, result_path_prefix, number_runs=number_runs, max_num_samples=max_num_samples, replace_existing_results=replace_existing_results, timeout_in_ms = timeout_in_ms)






