from amptorch.trainer import AtomsTrainer

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import os
import pdb

#structure for holding dataset and parameters
class dataset:
    def __init__(self, images, elements, atom_gaussians=None):
        self.images = images
        self.elements = elements
        self.atom_gaussians = atom_gaussians

#run k-fold cross validation with given mcsh params and return (train_mse, test_mse)
def evaluate_model(mcsh_group_params, cutoff, data, run_dir='./'):
    mcsh_params = {   "MCSHs": mcsh_group_params,
                      "atom_gaussians": data.atom_gaussians,
                      "cutoff": cutoff
                  }

    #setup for k-fold cross validation
    num_folds = 3

    #Separate data into k folds
    #The first n_samples % n_splits folds have size n_samples // n_splits + 1, 
    #other folds have size n_samples // n_splits, where n_samples is the number of samples.
    fold_indices = []
    num_larger_sets = len(data.images) % num_folds
    smaller_set_size = len(data.images) // num_folds
    larger_set_size = smaller_set_size + 1
    for i in range(num_folds):
        if i == 0:
            start_index = 0
        else:
            start_index = fold_indices[-1][1]

        if i < num_larger_sets:
            end_index = start_index + larger_set_size
        else:
            end_index = start_index + smaller_set_size

        fold_indices.append((start_index, end_index))

    #number of times to run CV
    num_cv_iters = 1 

    mse_test_list = []
    mse_train_list = []

    for _ in range(num_cv_iters):
        seed = np.random.randint(1, 100000)
        np.random.seed(seed)

        np.random.shuffle(data.images)
        true_energies = []
        for img in data.images:
            true_energies.append(img.get_potential_energy())

        #run k-fold cross validation
        for start_index, end_index in fold_indices:
            images_train = data.images[:start_index] + data.images[end_index:]
            images_test = data.images[start_index:end_index]

            true_energies_train = true_energies[:start_index] + true_energies[end_index:]
            true_energies_test = true_energies[start_index:end_index]

            config = {
                "model": {"get_forces": False, "num_layers": 3, "num_nodes": 20},
                "optim": {
                    "device": "cpu",
                    #"force_coefficient": 0.04,
                "force_coefficient": 0.0,
                    "lr": 1e-3,
                    "batch_size": 32,
                    "epochs": 1000,
                },
                "dataset": {
                    "raw_data": images_train,
                    "val_split": 0,
                    "elements": data.elements,
                    "fp_scheme": "mcsh",
                    "fp_params": mcsh_params,
                    "save_fps": False,
                },
                "cmd": {
                    "debug": False,
                    "run_dir": run_dir,
                    "seed": seed,
                    "identifier": "test",
                    "verbose": False,
                    "logger": False,
                },
            }

            trainer = AtomsTrainer(config)
            trainer.train()

            #test MSE
            predictions = trainer.predict(images_test)
            #true_energies = np.array([image.get_potential_energy() for image in images])
            pred_energies = np.array(predictions["energy"])
            curr_mse_test = np.mean((true_energies_test - pred_energies) ** 2)
            print("Test MSE:", curr_mse_test)
            mse_test_list.append(curr_mse_test)

            #train MSE
            predictions = trainer.predict(images_train)
            #true_energies = np.array([image.get_potential_energy() for image in images])
            pred_energies = np.array(predictions["energy"])
            curr_mse_train = np.mean((true_energies_train - pred_energies) ** 2)
            print("Train MSE:", curr_mse_train)
            mse_train_list.append(curr_mse_train)


    mse_test_avg = np.mean(mse_test_list)
    print("Avg test MSE: {}".format(mse_test_avg))

    mse_train_avg = np.mean(mse_train_list)
    print("Avg train MSE: {}".format(mse_train_avg))

    return mse_train_avg, mse_test_avg
