#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import json
import os
from os import listdir
from os.path import isfile, join
import pathlib
import csv
from os import listdir
from os.path import isfile, join
from pathlib import Path
import itertools
import re
import Scripts.Postprocessing as Postprocessing
import Scripts.ProblemGenerator as ProblemGenerator
import time


# In[2]:


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
    if os.path.exists(path + '/' + filename):
        try:
            os.remove(path + '/' + filename)
        except OSError:
            pass

    datapath = os.path.abspath(path)
    pathlib.Path(datapath).mkdir(parents=True, exist_ok=True) 
    
    datafile = os.path.abspath(path + '/' + filename)
    mode = 'a' if os.path.exists(datafile) else 'w'
    with open(datafile, mode) as file:
        json.dump(data, file)


# In[3]:


def optimise(card, pred, pred_sel, query_graph, allow_cross_products=True, timeout_in_ms = 60000):
    start = time.time()
    num_relations = len(card)
    result_register = {}
    relation_indices = np.arange(num_relations)
    for i in range(num_relations):
        result_register[frozenset([i])] = ([i], card[i])
    for s in range(1, num_relations+1):
        step_size = s - 1
        sub_trees = [h for h in result_register.keys() if len(h)==step_size]
        for sub_tree in sub_trees:
            sub_tree = frozenset(sub_tree)
            applicable_predicates = [pred_tuple for t in sub_tree for pred_tuple in query_graph if t in pred_tuple]
            for i in range(num_relations):
                if ((time.time() - start)*1000) > timeout_in_ms:
                    return [], None
                if i in sub_tree:
                    continue
                if len(applicable_predicates) != 0 and not allow_cross_products:
                    neighborhood_indices = [t for t in set(sum(applicable_predicates, ())) if t not in sub_tree]
                    if not i in neighborhood_indices:
                        continue
                sub_tree_result = result_register[sub_tree]
                if sub_tree_result is None:
                    continue
                sub_tree_order = sub_tree_result[0]
                new_tree_order = sub_tree_order.copy()
                new_tree_order.append(i)
                costs = Postprocessing.get_costs_for_leftdeep_tree(new_tree_order, card, pred, pred_sel, {}, verbose=False)
                new_sub_tree = frozenset(sorted(new_tree_order))
                if not new_sub_tree in result_register or costs < result_register[new_sub_tree][1]:
                    result_register[new_sub_tree] = [new_tree_order, costs]
    
    if ((time.time() - start)*1000) > timeout_in_ms:
        return [], None
    else:
        return result_register[frozenset(relation_indices)][0], result_register[frozenset(relation_indices)][1]


# In[4]:


def get_query_graph(query, num_rel):
    query_graph = []
    relations = query["relations"]
    card_dict = {}
    for relation in relations:
        card_dict[relation["name"]] = relation["cardinality"]
    
    joins = query["joins"]
    for join_entry in joins:
        join_relations = join_entry["relations"]
        for join_relation in join_relations:
            card1_index = list(card_dict.keys()).index(join_relations[0])
            card2_index = list(card_dict.keys()).index(join_relations[1])
            predicate = tuple(sorted((card1_index, card2_index)))
            if predicate not in query_graph:
                query_graph.append(predicate)
    return sorted(query_graph)

def conduct_benchmark_experiments(modes, problem_path_prefix, result_path_prefix, timeout_in_ms = 60000):
    benchmarks = os.listdir(path=problem_path_prefix + '/benchmark_queries')
    for benchmark in benchmarks:
        queries = os.listdir(path=problem_path_prefix + '/benchmark_queries/' + benchmark)
        for query in queries:
            query_number = int(query.split('q')[1])
            card, pred, pred_sel = ProblemGenerator.get_join_ordering_problem(problem_path_prefix + '/benchmark_queries/' + benchmark + '/' + query, generated_problems=True)
            raw_query = load_data(problem_path_prefix + '/benchmark_queries_raw/' + benchmark, query)
            query_graph = get_query_graph(raw_query, len(card))
            if len(card) <= 2:
                continue
            for mode in modes:
                if mode == "allow_cross_products":
                    start = time.time()
                    join_order, costs = optimise(card, pred, pred_sel, query_graph, allow_cross_products=True, timeout_in_ms=timeout_in_ms)
                    opt_time = time.time() - start
                    opt_time_in_ms = opt_time*1000
                    result = []
                    if len(join_order) > 0:
                        result.append({"time": opt_time_in_ms, "join_order": join_order, "costs:": int(costs)})
                    save_data(result, result_path_prefix + '/benchmark_queries/' + benchmark + '/' + query, 'dpsizelinearCP.json')
                elif mode == "ignore_cross_products":
                    start = time.time()
                    join_order, costs = optimise(card, pred, pred_sel, query_graph, allow_cross_products=False, timeout_in_ms=timeout_in_ms)
                    opt_time = time.time() - start
                    opt_time_in_ms = opt_time*1000
                    result = []
                    if len(join_order) > 0:
                        result.append({"time": opt_time_in_ms, "join_order": join_order, "costs:": int(costs)})
                    save_data(result, result_path_prefix + '/benchmark_queries/' + benchmark + '/' + query, 'dpsizelinearNCP.json')
                else:
                    continue
                    
def conduct_synthetic_experiments(modes, problem_path_prefix, result_path_prefix, timeout_in_ms = 60000):
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

                result_path = result_path_prefix + '/synthetic_queries/' + problem_path_main
                for mode in modes:
                    if mode == "allow_cross_products":
                        start = time.time()
                        join_order, costs = optimise(card, pred, pred_sel, allow_cross_products=True, timeout_in_ms=timeout_in_ms)
                        opt_time = time.time() - start
                        opt_time_in_ms = opt_time*1000
                        result = []
                        if len(join_order) > 0:
                            result.append({"time": opt_time_in_ms, "join_order": join_order, "costs:": int(costs)})
                        save_data(result, result_path, 'dpsizelinearCP.json')
                    elif mode == "ignore_cross_products":
                        start = time.time()
                        join_order, costs = optimise(card, pred, pred_sel, allow_cross_products=False, timeout_in_ms=timeout_in_ms)
                        opt_time = time.time() - start
                        opt_time_in_ms = opt_time*1000
                        result = []
                        if len(join_order) > 0:
                            result.append({"time": opt_time_in_ms, "join_order": join_order, "costs:": int(costs)})
                        save_data(result, result_path, 'dpsizelinearNCP.json')
                    else:
                        continue

if __name__ == "__main__":
    modes = ['ignore_cross_products']
    problem_path_prefix = 'ExperimentalAnalysis/Problems'
    result_path_prefix = 'ExperimentalAnalysis/ResultsBenchmarks'
    conduct_benchmark_experiments(modes, problem_path_prefix, result_path_prefix, timeout_in_ms = 60000)

