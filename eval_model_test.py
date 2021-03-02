import os

import numpy as np

from ase import Atoms
from ase.calculators.emt import EMT

import evaluate_mcsh_model



#setup dataset
np.random.seed(3)
distances = np.linspace(2, 5, 100)
images = []
for i in range(len(distances)):
    l = distances[i]
    image = Atoms(
        "CuCO",
        [
            (-l * np.sin(0.65), l * np.cos(0.65), np.random.uniform(low=-4.0, high=4.0)),
            (0, 0, 0),
            (l * np.sin(0.65), l * np.cos(0.65), np.random.uniform(low=-4.0, high=4.0))
        ],
    )
    image.set_cell([10, 10, 10])
    image.wrap(pbc=True)
    image.set_calculator(EMT())
    images.append(image)

elements = ["Cu","C","O"]
dir_prefix = "/storage/home/hpaceice1/plai30/sandbox"
atom_gaussians = {"C": os.path.join(dir_prefix, "config/MCSH_potential/C_coredensity_5.g"),
                  "O": os.path.join(dir_prefix, "config/MCSH_potential/O_totaldensity_7.g"),
                  "Cu": os.path.join(dir_prefix, "config/MCSH_potential/Cu_totaldensity_5.g")}
data = evaluate_mcsh_model.dataset(images, elements, atom_gaussians)

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
base_order = 1 #number of orders to include by default
base_params = {str(i): groups_by_order[i] for i in range(base_order + 1)}

#run model evaluation
train_mse, test_mse = evaluate_mcsh_model.evaluate_model(base_params, cutoff, data)

#print results
print("CV error: {}".format(test_mse))

