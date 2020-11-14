import numpy as np
import pandas as pd

import copy
import os
import pdb

import evaluate_mcsh_model

OUTPUT_DIR = "D:\\Work\\sandbox\\vip\\outputs"

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
base_order = 3 #number of orders to include by default
base_group_params = {str(i): groups_by_order[i] for i in range(base_order + 1)}

#get baseline performance
print("Testing base params: {}".format(base_group_params))
base_train_mse, base_test_mse = evaluate_mcsh_model.evaluate_model(base_group_params, cutoff)
# base_train_mse = 0.32747516648021247
# base_test_mse = 0.3675139149938907
print("Base test MSE: {}".format(base_test_mse))

#stop criteria
stop_improvement_pct = 0.05
max_features = 50
num_features_added = 0
prev_test_mse = base_test_mse
prev_group_params = copy.deepcopy(base_group_params)

MSEs = [base_test_mse]
orders_added = [-1]
groups_added = [-1]

#groups to be removed from consideration, organized as {order: set(groups)}
removed_groups = {}
remove_improvement_pct = -0.5

print("Forward selection params: stop_improvement_pct={}, max_features={}, remove_improvement_pct={}".format(
        stop_improvement_pct, max_features, remove_improvement_pct))

#perform forward selection with early dropping
while True:
    curr_min_test_mse = 1000000.
    curr_best_order = -1
    curr_best_group = -1
    curr_best_params = None

    for order, order_params in groups_by_order.items():
        if order <= base_order:
            continue

        order_str = str(order)

        for group in order_params["groups"]:
            #check if this group has been removed
            if order in removed_groups:
                if group in removed_groups[order]:
                    continue

            group_params_candidate = copy.deepcopy(prev_group_params)

            #skip if this group is already included
            if order_str in prev_group_params:
                if group in prev_group_params[order_str]["groups"]:
                    continue
            else:
                #if this is the first group in this order, add the order to params
                group_params_candidate[order_str] = {"groups": [], "sigmas": sigmas}

            group_params_candidate[order_str]["groups"].append(group)
            group_params_candidate[order_str]["groups"].sort()

            print("Testing order {}, group {}".format(order, group))
            print("MCSH group params: {}".format(group_params_candidate))
            train_mse, test_mse = evaluate_mcsh_model.evaluate_model(group_params_candidate, cutoff)

            if test_mse < curr_min_test_mse:
                curr_min_test_mse = test_mse
                curr_best_order = order
                curr_best_group = group
                curr_best_params = copy.deepcopy(group_params_candidate)

            #check if this group should be removed
            curr_improvement_pct = (prev_test_mse - test_mse) / prev_test_mse
            if curr_improvement_pct < remove_improvement_pct:
                print("Improvement for order {}, group {} is less than {}, removing from consideration".format(
                    order, group, remove_improvement_pct))

                if order in removed_groups:
                    removed_groups[order].add(group)
                else:
                    removed_groups[order] = {group}

    best_improvement_pct = (prev_test_mse - curr_min_test_mse) / prev_test_mse
    print("Adding order {}, group {} improved test MSE by {} pct ({} to {})".format(
        curr_best_order, curr_best_group, best_improvement_pct, prev_test_mse, curr_min_test_mse))

    #check for stop criteria
    if best_improvement_pct > stop_improvement_pct:
        prev_test_mse = curr_min_test_mse
        prev_group_params = copy.deepcopy(curr_best_params)

        MSEs.append(curr_min_test_mse)
        orders_added.append(curr_best_order)
        groups_added.append(curr_best_group)

        num_features_added += 1

        #write results to file (overwrite on each iteration)
        results = pd.DataFrame(data={"order": orders_added, 
                                     "group": groups_added, 
                                     "test_mse": MSEs, 
                                     "iteration": range(len(MSEs))})
        results.to_csv(os.path.join(OUTPUT_DIR, "forward_selection_results.csv"))

        if num_features_added >= max_features:
            print("Max number of additional features ({}) reached, stopping".format(num_features_added))
            break

    else:
        print("Best improvement was less than {} pct, stopping".format(stop_improvement_pct))
        break