import numpy as np
import pandas as pd

import copy
import json
import os
import pathlib
import pdb
import shutil
import subprocess
import time

import evaluate_mcsh_model
from utils import pace_utils

def forward_selection(output_dir, data):
    cutoff = 8
    sigmas = np.logspace(np.log10(0.05), np.log10(1.0), num=5)
    groups_by_order = {0: {"groups": [1], "sigmas": sigmas},
                       1: {"groups": [1], "sigmas": sigmas},
                       2: {"groups": [1,2], "sigmas": sigmas},
                       3: {"groups": [1,2,3], "sigmas": sigmas},
                       4: {"groups": [1,2,3,4], "sigmas": sigmas},
                       5: {"groups": [1,2,3,4,5], "sigmas": sigmas},
                       6: {"groups": [1,2,3,4,5,6,7], "sigmas": sigmas},
                       7: {"groups": [1,2,3,4,5,6,7,8], "sigmas": sigmas},
                       8: {"groups": [1,2,3,4,5,6,7,8,9,10], "sigmas": sigmas},
                       9: {"groups": [1,2,3,4,5,6,7,8,9,10,11,12], "sigmas": sigmas}}


    #setup baseline MCSH params
    base_order = 0 #number of orders to include by default
    base_params = {str(i): groups_by_order[i] for i in range(base_order + 1)}

    #get baseline performance
    print("Testing base params: {}".format(base_params))
    base_train_mse, base_test_mse = evaluate_mcsh_model.evaluate_model(base_params, cutoff, data)
    print("Base test MSE: {}".format(base_test_mse))

    stop_improvement_pct = 0.05
    max_features = 50
    num_features_added = 0
    prev_test_mse = base_test_mse
    prev_group_params = copy.deepcopy(base_params)

    MSEs = []
    orders_added = []

    #orders to be removed from consideration
    removed_orders = set()
    remove_improvement_pct = -2.0

    print("Forward selection params: stop_improvement_pct={}, max_features={}, remove_improvement_pct={}".format(
            stop_improvement_pct, max_features, remove_improvement_pct))

    #perform forward selection with early dropping
    while True:
        curr_min_test_mse = 1000000.
        curr_best_order = -1
        curr_best_params = None

        for order, order_params in groups_by_order.items():

            #check if this order has been removed
            if order in removed_orders:
                continue

            order_str = str(order)
            
            #check if this order has already been included
            if order_str in prev_group_params:
                continue

            group_params_candidate = copy.deepcopy(prev_group_params)
            group_params_candidate[order_str] = groups_by_order[order]


            print("Testing order {}".format(order))
            print("MCSH group params: {}".format(group_params_candidate))
            train_mse, test_mse = evaluate_mcsh_model.evaluate_model(group_params_candidate, cutoff, model)

            if test_mse < curr_min_test_mse:
                curr_min_test_mse = test_mse
                curr_best_order = order
                curr_best_params = copy.deepcopy(group_params_candidate)

            #check if this group should be removed
            curr_improvement_pct = (prev_test_mse - test_mse) / prev_test_mse
            if curr_improvement_pct < remove_improvement_pct:
                print("Improvement for order {} is less than {}, removing from consideration".format(
                    order, remove_improvement_pct))

                removed_orders.add(order)



        best_improvement_pct = (prev_test_mse - curr_min_test_mse) / prev_test_mse
        print("Adding order {} improved test MSE by {} pct ({} to {})".format(
            curr_best_order, best_improvement_pct, prev_test_mse, curr_min_test_mse))

        #check for stop criteria
        if best_improvement_pct > stop_improvement_pct:
            prev_test_mse = curr_min_test_mse
            prev_group_params = copy.deepcopy(curr_best_params)

            MSEs.append(curr_min_test_mse)
            orders_added.append(curr_best_order)

            num_features_added += 1

            #write results to file (overwrite on each iteration)
            results = pd.DataFrame(data={"order": orders_added, 
                                         "test_mse": MSEs, 
                                         "iteration": range(len(MSEs))})
            results.to_csv(os.path.join(output_dir, "forward_selection_results.csv"))

            if num_features_added >= max_features:
                print("Max number of additional features ({}) reached, stopping".format(num_features_added))
                break

        else:
            print("Best improvement was less than {} pct, stopping".format(stop_improvement_pct))
            break


def backward_elimination(output_dir, data, enable_parallel, parallel_workspace=None):
    cutoff = 8
    sigmas = np.logspace(np.log10(0.05), np.log10(1.0), num=5)
    groups_by_order = {0: {"groups": [1], "sigmas": sigmas},
                       1: {"groups": [1], "sigmas": sigmas},
                       2: {"groups": [1,2], "sigmas": sigmas},
                       3: {"groups": [1,2,3], "sigmas": sigmas},
                       4: {"groups": [1,2,3,4], "sigmas": sigmas},
                       5: {"groups": [1,2,3,4,5], "sigmas": sigmas},
                       6: {"groups": [1,2,3,4,5,6,7], "sigmas": sigmas},
                       7: {"groups": [1,2,3,4,5,6,7,8], "sigmas": sigmas},
                       8: {"groups": [1,2,3,4,5,6,7,8,9,10], "sigmas": sigmas},
                       9: {"groups": [1,2,3,4,5,6,7,8,9,10,11,12], "sigmas": sigmas}}

    #setup baseline MCSH params
    base_order = 9 #number of orders to include by default
    base_params = {str(i): groups_by_order[i] for i in range(base_order + 1)}

    #get baseline performance
    print("Testing base params: {}".format(base_params))
    #base_train_mse, base_test_mse = evaluate_mcsh_model.evaluate_model(base_params, cutoff, data)
    base_train_mse = 1.
    base_test_mse = 1.
    print("Base test MSE: {}".format(base_test_mse))

    stop_change_pct = 0.1
    prev_test_mse = base_test_mse
    prev_group_params = copy.deepcopy(base_params)

    MSEs = [base_test_mse]
    orders_removed = [-1]

    print("Backward elimination params: stop_change_pct={}".format(
            stop_change_pct))

    #setup necessary directories, write dataset to file if using Pace
    if enable_parallel:
        print("Parallel processing enabled. Initializing workspace")

        workspace_subdirs = []
        
        config_path = pathlib.Path(parallel_workspace) / "config"
        workspace_subdirs.append(config_path)

        pbs_path = pathlib.Path(parallel_workspace) / "pbs"
        workspace_subdirs.append(pbs_path)

        temp_output_path = pathlib.Path(parallel_workspace) / "output"
        workspace_subdirs.append(temp_output_path)

        training_path = pathlib.Path(parallel_workspace) / "training"
        workspace_subdirs.append(training_path)

        for subdir in workspace_subdirs:
            if subdir.exists() and subdir.is_dir():
                shutil.rmtree(subdir)
            
            subdir.mkdir(parents=True, exist_ok=False)

        #write dataset

    #perform backward elimination
    while True:
        curr_min_test_mse = 1000000.
        curr_best_order = -1
        curr_best_params = None

        if enable_parallel:
            candidate_orders = []
            candidate_params = []
            
            print("Creating configs for processing on Pace")
            for order, order_params in prev_group_params.items():
                group_params_candidate = copy.deepcopy(prev_group_params)
                order_str = str(order)
                del group_params_candidate[order_str]

                #create model configs for each model to test
                curr_config = {}
                curr_config["id"] = order
                curr_config["cutoff"] = cutoff
                curr_config["groups_by_order"] = {str(k): {"groups": v["groups"]} for (k, v) in group_params_candidate.items()}
                curr_config["sigmas"] = list(sigmas)

                curr_config_file = config_path / "config_{}.json".format(order)
                json.dump(curr_config, open(curr_config_file, "w+"))

                #create pace pbs files
                model_eval_script = "/storage/home/hpaceice1/plai30/sandbox/mcsh_exploration/run_eval_test.py"
                data_file = "/storage/home/hpaceice1/plai30/sandbox/data/test/test_data.p" #TODO: remove hardcoded dataset
                command_str = "python {} --workspace {} --job_id {} --data {}".format(model_eval_script, parallel_workspace, order, data_file) 
                pbs_file = pace_utils.create_pbs(pbs_path, order_str, command_str, time="00:10:00")

                #submit job on pace
                subprocess.run(["qsub", pbs_file])
                
                candidate_orders.append(order)
                candidate_params.append(group_params_candidate)

            #collect results
            for i in range(len(candidate_orders)):
                order = candidate_orders[i]
                group_params_candidate = candidate_params[i]

                curr_output_file = temp_output_path / "output_{}.json".format(order)
                while not curr_output_file.exists():
                    print("results for job {} not ready. Sleeping for 20s".format(order))
                    time.sleep(20)

                result_dict = json.load(open(curr_output_file, "r"))
                test_mse = result_dict["avg_test_mse"]

                if test_mse < curr_min_test_mse:
                    curr_min_test_mse = test_mse
                    curr_best_order = order
                    curr_best_params = copy.deepcopy(group_params_candidate)

            #clear workspace
            for subdir in workspace_subdirs:
                if subdir.exists() and subdir.is_dir():
                    shutil.rmtree(subdir)

                subdir.mkdir(parents=True, exist_ok=False)
                
        else:
            #run locally sequentially
            for order, order_params in prev_group_params.items():

                group_params_candidate = copy.deepcopy(prev_group_params)
                order_str = str(order)
                del group_params_candidate[order_str]

                print("Testing removing order {}".format(order))
                print("MCSH group params: {}".format(group_params_candidate))
                train_mse, test_mse = evaluate_mcsh_model.evaluate_model(group_params_candidate, cutoff, data)

                if test_mse < curr_min_test_mse:
                    curr_min_test_mse = test_mse
                    curr_best_order = order
                    curr_best_params = copy.deepcopy(group_params_candidate)



        max_change_pct = (curr_min_test_mse - prev_test_mse) / prev_test_mse
        print("Best change: removing order {} changed test MSE by {} pct ({} to {})".format(
            curr_best_order, max_change_pct, prev_test_mse, curr_min_test_mse))
        print("Params for best change: {}".format(curr_best_params))

        #check for stop criteria
        if max_change_pct < stop_change_pct:
            prev_test_mse = curr_min_test_mse
            prev_group_params = copy.deepcopy(curr_best_params)

            MSEs.append(curr_min_test_mse)
            orders_removed.append(curr_best_order)

            #write results to file (overwrite on each iteration)
            results = pd.DataFrame(data={"order": orders_removed, 
                                         "test_mse": MSEs, 
                                         "iteration": range(len(MSEs))})
            results.to_csv(os.path.join(output_dir, "backward_elimination_results.csv"))

        else:
            print("Best change was less than {} pct, stopping".format(stop_change_pct))
            break
