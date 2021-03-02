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

from model_eval import constants, model_evaluation, utils

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


def backward_elimination(output_dir, data, enable_parallel, parallel_workspace=None, seed=None):
    cutoff = 8
    sigmas = (np.logspace(np.log10(0.05), np.log10(1.0), num=5)).tolist()
    groups_by_order = {0: {"groups": [1]},
                       1: {"groups": [1]},
                       2: {"groups": [1,2]},
                       3: {"groups": [1,2,3]},
                       4: {"groups": [1,2,3,4]},
                       5: {"groups": [1,2,3,4,5]},
                       6: {"groups": [1,2,3,4,5,6,7]},
                       7: {"groups": [1,2,3,4,5,6,7,8]},
                       8: {"groups": [1,2,3,4,5,6,7,8,9,10]},
                       9: {"groups": [1,2,3,4,5,6,7,8,9,10,11,12]}}

    #setup baseline MCSH params
    base_order = 9 #number of orders to include by default
    base_group_params = {str(i): groups_by_order[i] for i in range(base_order + 1)}

    base_params = model_evaluation.get_model_eval_params("base", "mcsh", "k_fold_cv", eval_num_folds=5, eval_cv_iters=3, 
                                                         cutoff=cutoff, mcsh_groups=base_group_params, sigmas=sigmas, seed=seed)

    #get baseline performance
    print("Testing base params: {}".format(base_params))
    base_train_mse, base_test_mse = model_evaluation.evaluate_model(base_params, data)
    print("Base test MSE: {}".format(base_test_mse))

    stop_change_pct = 0.15
    prev_test_mse = base_test_mse
    prev_group_params = copy.deepcopy(base_group_params)

    MSEs = [base_test_mse]
    orders_removed = [-1]

    print("Backward elimination params: stop_change_pct={}".format(
            stop_change_pct))

    #perform backward elimination
    while True:
        curr_min_test_mse = 1000000.
        curr_best_order = -1
        curr_best_group_params = None

        candidate_orders = []
        candidate_params = []
        
        print("Creating configs for processing on Pace")
        for order, order_params in prev_group_params.items():
            group_params_candidate = copy.deepcopy(prev_group_params)
            order_str = str(order)
            del group_params_candidate[order_str]

            eval_params_candidate = copy.deepcopy(base_params)
            eval_params_candidate[constants.CONFIG_JOB_NAME] = str(order)
            eval_params_candidate[constants.CONFIG_MCSH_GROUPS] = group_params_candidate
            
            candidate_orders.append(order)
            candidate_params.append(eval_params_candidate)

        results = model_evaluation.evaluate_models(data, config_dicts=candidate_params, 
                                                   enable_parallel=enable_parallel, workspace=parallel_workspace,
                                                   time_limit="00:40:00", mem_limit=2, conda_env="amptorch")

        for i in range(len(candidate_orders)):
            curr_test_mse = results[i].test_error
            curr_order = candidate_orders[i]
            curr_params = candidate_params[i]

            if curr_test_mse < curr_min_test_mse:
                curr_min_test_mse = curr_test_mse
                curr_best_order = curr_order
                curr_best_group_params = copy.deepcopy(curr_params[constants.CONFIG_MCSH_GROUPS])

        max_change_pct = (curr_min_test_mse - prev_test_mse) / prev_test_mse
        print("Best change: removing order {} changed test MSE by {} pct ({} to {})".format(
            curr_best_order, max_change_pct, prev_test_mse, curr_min_test_mse))
        print("Params for best change: {}".format(curr_best_group_params))

        #check for stop criteria
        if max_change_pct < stop_change_pct:
            
            if max_change_pct < 0.:
                prev_test_mse = curr_min_test_mse
    
            prev_group_params = copy.deepcopy(curr_best_group_params)

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
