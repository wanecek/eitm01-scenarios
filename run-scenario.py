CM_BASEPATH = "../cibusmod"

import sys
import os

sys.path.insert(0, os.path.join(os.getcwd(), CM_BASEPATH))

import CIBUSmod as cm
import CIBUSmod.utils.plot as plot

import time
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import cvxpy

from CIBUSmod.utils.misc import inv_dict, aggregate_data_coords_pair
from CIBUSmod.optimisation.indexed_matrix import IndexedMatrix
from itertools import product

# Create session
session = cm.Session(
    name="ww_scenarios",
    data_path=CM_BASEPATH + "/data",
    data_path_default=CM_BASEPATH + "/data/default",
    data_path_scenarios="../scenarios",
)

# Load and apply scenario
session.add_scenario("base", years=[2020], pars="all", scenario_workbooks="base")

retrievers = {
    "Regions": cm.ParameterRetriever("Regions"),
    "DemandAndConversions": cm.ParameterRetriever("DemandAndConversions"),
    "CropProduction": cm.ParameterRetriever("CropProduction"),
    "FeedMgmt": cm.ParameterRetriever("FeedMgmt"),
    "GeoDistributor": cm.ParameterRetriever("GeoDistributor"),
}

for par in retrievers.values():
    par.update_all_parameter_values(**session["base"], year=2020)

# Instatiate Regions
regions = cm.Regions(
    par=retrievers["Regions"],
)

# Instantiate DemandAndConversions
demand = cm.DemandAndConversions(
    par=retrievers["DemandAndConversions"],
)

# Instantiate CropProduction
crops = cm.CropProduction(
    par=retrievers["CropProduction"], index=regions.data_attr.get("x0_crops").index
)

# Instantiate AnimalHerds
# Each AnimalHerd object is stored in an indexed pandas.Series
herds = cm.make_herds(regions)

# Instantiate feed management
feed_mgmt = cm.FeedMgmt(
    herds=herds,
    par=retrievers["FeedMgmt"],
)

# Instantiate geo distributor
optproblem = cm.FeedDistributor(
    regions=regions,
    demand=demand,
    crops=crops,
    herds=herds,
    feed_mgmt=feed_mgmt,
    par=retrievers["GeoDistributor"],
)

self = optproblem

cm.ParameterRetriever.update_all_parameter_values()
cm.ParameterRetriever.update_relation_tables()

cm.ParameterRetriever.update_all_parameter_values(**session["base"], year=2020)

regions.calculate()
demand.calculate()
crops.calculate()
for h in herds:
    h.calculate()

self.make(use_cons=[1, 2, 3, 4, 5, 6, 11, 12, 13, 14])

###

PROTEIN_CONTENTS = {
    "Peas (add)": 12.2,
    "Wheat (add)": 5.20,
    "meat": 100.0,
    "milk": 10.0,
}


def make_protein_mask_ani():
    RELEVANT_ANIMAL_PRODUCTS = ["meat", "milk"]

    # Get row index from animal product demand vector (ps,sp,ap)
    row_idx = pd.MultiIndex.from_tuples(
        filter(
            lambda tup: tup[1] == "cattle" and tup[2] in ["meat", "milk"],
            self.D_idx["ani"].values,
        ),
        names=self.D_idx["ani"].names,
    )

    # Get col index from animal herds (sp,br,ps,ss,re)
    col_idx = self.x_idx["ani"]

    # To store data and corresponding row/col numbers for constructing matrix
    val = []
    row_nr = []
    col_nr = []

    # Go through animal herds
    for herd in self.herds:
        sp = herd.species
        br = herd.breed
        ps = herd.prod_system
        ss = herd.sub_system

        if sp != "cattle":
            continue

        def get_uniq(col):
            return herd.data_attr.get("production").columns.unique(col)

        # Get all animal products that we are concerned with
        aps = set(get_uniq("animal_prod")) & set(RELEVANT_ANIMAL_PRODUCTS)
        opss = get_uniq("prod_system")

        for ap, ops in product(aps, opss):
            # Ensure it's
            if (ops, herd.species, ap) not in row_idx:
                continue

            # Get production of animal product (ap) from output production system (ops) per head
            # of defining animal of species (sp) and breed (br) in production system (ps), sub system (ss)
            # and region (re)
            res = (
                herd.data_attr.get("production")
                .loc[:, (ops, slice(None), ap)]
                .sum(axis=1)
            ) * PROTEIN_CONTENTS[ap]

            if all(res == 0):
                continue

            val.extend(res)
            col_nr.extend([col_idx.get_loc((sp, br, ps, ss, re)) for re in res.index])
            row_nr.extend(np.zeros(len(res)))

    # Aggregate data_coords_pair to ensure that any overlapping values are summed rather than replace each other
    val, (row_nr, col_nr) = aggregate_data_coords_pair(val, row_nr, col_nr)

    # Create Compressed Sparse Column matrix
    return scipy.sparse.coo_array(
        (val, (row_nr, col_nr)), shape=(1, len(col_idx))
    ).tocsc()


def make_protein_mask_crp():
    wheat_locs = self.x_idx["crp"].get_locs(("Wheat (add)",))
    peas_locs = self.x_idx["crp"].get_locs(("Peas (add)",))

    val = ([PROTEIN_CONTENTS["Wheat (add)"]] * len(wheat_locs)) + (
        [PROTEIN_CONTENTS["Peas (add)"]] * len(peas_locs)
    )
    col_nr = [*wheat_locs, *peas_locs]
    row_nr = np.zeros(len(val))

    return scipy.sparse.coo_array(
        (val, (row_nr, col_nr)), shape=(1, len(self.x_idx["crp"]))
    ).tocsc()


def make_protein_mask():
    A_ani = make_protein_mask_ani()
    A_crp = make_protein_mask_crp()
    A_fds = scipy.sparse.csc_matrix((1, len(self.x_idx["fds"])))

    return scipy.sparse.hstack([A_ani, A_crp, A_fds], format="csc")


def construct_problem(objective_fn):
    n = (
        len(self.x_idx_short["ani"])
        + len(self.x_idx_short["crp"])
        + len(self.x_idx_short["fds"])
    )
    x = cvxpy.Variable(n, nonneg=True)

    # Append constraints
    constraints = [
        cm.optimisation.utils.make_cvxpy_constraint(cons, x)
        for cons in self.constraints.values()
    ]

    # Define problem
    return cvxpy.Problem(objective=objective_fn(x), constraints=constraints)


# protein_map = make_protein_mask()
# self.problem = construct_problem(lambda x: cvxpy.Maximize(protein_map @ x))

self.solve(
    verbose=True,
    apply_solution=False,
    solver_settings=[
        {
            "solver": "GUROBI",
            "verbose": True,
            "reoptimize": True,
        }
    ],
)

for k in ["crp", "ani", "fds"]:
    self.x[k].to_pickle(f"x.{k}.pkl")

