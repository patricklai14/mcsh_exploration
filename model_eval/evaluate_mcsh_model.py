from amptorch.trainer import AtomsTrainer

import matplotlib.pyplot as plt
import numpy as np

import pdb

import constants

#structure for holding dataset and parameters
class dataset:
    def __init__(self, elements, train_images, test_images=None, atom_gaussians=None):
        self.elements = elements
        self.train_images = train_images
        self.test_images = test_images
        self.atom_gaussians = atom_gaussians

class evaluation_params:
    def __init__(self, config):
        self.validate_config(config)
        self.set_default_params()
        self.set_config_params(config)

    def validate_config(self, config):
        required_fields = [constants.CONFIG_CUTOFF, 
                           constants.CONFIG_FP_TYPE, 
                           constants.CONFIG_EVAL_TYPE]

        for field in required_fields:
            if field not in config:
                raise RuntimeError("required field {} not in config".format(field))

        if config[constants.CONFIG_FP_TYPE] == "mcsh":
            if constants.CONFIG_GROUPS_BY_ORDER not in config or constants.CONFIG_SIGMAS not in config:
                raise RuntimeError("incomplete information in config for MCSH")

                #TODO: validate mcsh parameters

        elif config[constants.CONFIG_FP_TYPE] == "bp":
            if constants.BP_PARAMS not in config:
                raise RuntimeError("bp_params required in config for BP")

            #TODO: validate bp parameters
        
        else:
            raise RuntimeError("invalid fingerprint type: {}".format(config[constants.CONFIG_FP_TYPE]))

    def set_default_params(self):
        self.params = {constants.CONFIG_NN_LAYERS: constants.DEFAULT_NN_LAYERS,
                       constants.CONFIG_NN_NODES: constants.DEFAULT_NN_NODES,
                       constants.CONFIG_NN_LR: constants.DEFAULT_NN_LR,
                       constants.CONFIG_NN_BATCH_SIZE: constants.DEFAULT_NN_BATCH_SIZE,
                       constants.CONFIG_NN_EPOCHS: constants.DEFAULT_NN_EPOCHS,
                       constants.CONFIG_EVAL_NUM_FOLDS: constants.DEFAULT_EVAL_NUM_FOLDS,
                       constants.CONFIG_EVAL_CV_ITERS: constants.DEFAULT_EVAL_CV_ITERS,
                       constants.CONFIG_RAND_SEED: constants.DEFAULT_RAND_SEED}

    def set_config_params(self, config):
        for key, value in config.items():
            if key == constants.CONFIG_GROUPS_BY_ORDER:
                mcsh_group_params = value
                for order, group_params in mcsh_group_params.items():
                    group_params[constants.CONFIG_SIGMAS] = np.array(config[constants.CONFIG_SIGMAS])

                self.params[constants.PARAM_MCSH_GROUP_PARAMS] = mcsh_group_params
                continue

            self.params[key] = value

#evaluate model with a single train/test split
def evaluate_model_one_split(eval_params, data):
    if eval_params.params[constants.CONFIG_FP_TYPE] == "mcsh":
        fp_scheme = "mcsh"
        fp_params = {"MCSHs": eval_params.params[constants.PARAM_MCSH_GROUP_PARAMS],
                     "atom_gaussians": data.atom_gaussians,
                     "cutoff": eval_params.params[constants.CONFIG_CUTOFF]
                    }

    else:
        fp_scheme = "bp"
        fp_params = eval_params.params[constants.CONFIG_BP_PARAMS]

    config = {
        "model": {"get_forces": False, 
                  "num_layers": eval_params.params[constants.CONFIG_NN_LAYERS], 
                  "num_nodes": eval_params.params[constants.CONFIG_NN_NODES]},
        "optim": {
            "device": "cpu",
            "force_coefficient": 0.0,
            "lr": eval_params.params[constants.CONFIG_NN_LR],
            "batch_size": eval_params.params[constants.CONFIG_NN_BATCH_SIZE],
            "epochs": eval_params.params[constants.CONFIG_NN_EPOCHS],
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
            "seed": eval_params.params[constants.CONFIG_RAND_SEED],
            "identifier": "test",
            "verbose": False,
            "logger": False,
        },
    }

    trainer = AtomsTrainer(config)
    trainer.train()

    #test MSE
    predictions = trainer.predict(data.test_images)
    true_energies_test = np.array([image.get_potential_energy() for image in images])
    pred_energies = np.array(predictions["energy"])
    curr_mse_test = np.mean((true_energies_test - pred_energies) ** 2)
    print("Test MSE:", curr_mse_test)

    #train MSE
    predictions = trainer.predict(data.train_images)
    true_energies_train = np.array([image.get_potential_energy() for image in images])
    pred_energies = np.array(predictions["energy"])
    curr_mse_train = np.mean((true_energies_train - pred_energies) ** 2)
    print("Train MSE:", curr_mse_train)

    return curr_mse_train, curr_mse_test

#run model evaluation with given params and return (train_mse, test_mse)
def evaluate_model(eval_config, data, run_dir='./'):
    eval_params = evaluation_params(eval_config)
    np.random.seed(eval_params.params[constants.CONFIG_RAND_SEED])

    if eval_config.params[constants.CONFIG_EVAL_TYPE] == "k_fold_cv":

        #setup for k-fold cross validation
        num_folds = eval_params.params[constants.CONFIG_EVAL_NUM_FOLDS]

        #Separate data into k folds
        #The first n_samples % n_splits folds have size n_samples // n_splits + 1, 
        #other folds have size n_samples // n_splits, where n_samples is the number of samples.
        fold_indices = []
        num_larger_sets = len(data.train_images) % num_folds
        smaller_set_size = len(data.train_images) // num_folds
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
        num_cv_iters = eval_params.params[constants.CONFIG_EVAL_CV_ITERS]

        mse_test_list = []
        mse_train_list = []

        for _ in range(num_cv_iters):
            np.random.shuffle(data.train_images)
            true_energies = []
            for img in data.train_images:
                true_energies.append(img.get_potential_energy())

            #run k-fold cross validation
            for start_index, end_index in fold_indices:
                images_train = data.train_images[:start_index] + data.train_images[end_index:]
                images_test = data.train_images[start_index:end_index]

                curr_data = dataset(data.elements, images_train, images_test, data.atom_gaussians)
                curr_mse_train, curr_mse_test = evaluate_model_one_split(eval_params, curr_data)
                
                mse_test_list.append(curr_mse_test)
                mse_train_list.append(curr_mse_train)

        mse_test_avg = np.mean(mse_test_list)
        print("Avg test MSE: {}".format(mse_test_avg))

        mse_train_avg = np.mean(mse_train_list)
        print("Avg train MSE: {}".format(mse_train_avg))

        return mse_train_avg, mse_test_avg


    else:
        #simple train on training set, test on test set
        mse_train, mse_test = evaluate_model_one_split(eval_params, data)
        return mse_train, mse_test