import numpy as np

import argparse
import json
import os
import pathlib
import pdb
import pickle

import evaluate_mcsh_model
import selection_methods

def main():
    parser = argparse.ArgumentParser(description="Run model evaluation")
    parser.add_argument("--workspace", type=str, required=True, 
                        help="top-level workspace")
    parser.add_argument("--job_id", type=str, required=True,
                        help="name of job")
    parser.add_argument("--data", type=str, required=True,
                        help="location of dataset file")

    args = parser.parse_args()
    args_dict = vars(args)

    workspace_path = pathlib.Path(args_dict["workspace"])
    job_id = args_dict["job_id"]

    #load dataset
    data_filename = args_dict["data"]
    dataset = pickle.load(open(data_filename, "rb"))

    #load model config
    config_file = workspace_path / "config/config_{}.json".format(job_id)
    config = json.load(open(config_file, "r"))

    cutoff = config["cutoff"]
    order_params = config["groups_by_order"]
    for order, params in order_params.items():
        params["sigmas"] = np.array(config["sigmas"])

    run_dir = workspace_path / "training/{}".format(job_id)
    run_dir.mkdir(parents=True, exist_ok=False)

    #get model performance
    print("Evaluating with params: {}".format(order_params))
    train_mse, test_mse = evaluate_mcsh_model.evaluate_model(order_params, cutoff, dataset, run_dir)
    print("Test MSE: {}".format(test_mse))

    #write result to file
    output_path = workspace_path / "output" / "output_{}.json".format(config["id"])
    json.dump({"avg_train_mse": train_mse, "avg_test_mse": test_mse}, open(output_path, "w+"))

if __name__ == "__main__":
    main()

