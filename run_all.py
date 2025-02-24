CM_BASEPATH = "../cibusmod"

import sys
import os

sys.path.insert(0, os.path.join(os.getcwd(), CM_BASEPATH))

import CIBUSmod as cm

import numpy as np
import pandas as pd
import scipy
import cvxpy

from CIBUSmod.optimisation.indexed_matrix import IndexedMatrix
from CIBUSmod.optimisation.utils import make_cvxpy_constraint
from itertools import product

def aggregate_data_coords_pair(values, row_i, col_i):
    if (len(values) != len(row_i)) or (len(values) != len(col_i)):
        raise ValueError(
            "Length mismatch. Lists values, row_i and col_i did not all have the same length."
        )
    if len(values) == 0:
        return ([], ([], []))
    # Combine rows, cols, and values into a single array for aggregation
    data = np.vstack((row_i, col_i, values)).T
    # Aggregate using numpy
    unique_coords, indices = np.unique(data[:, :2], axis=0, return_inverse=True)
    aggregated_values = np.zeros(len(unique_coords))
    np.add.at(aggregated_values, indices, data[:, 2])
    # Split unique coordinates back into rows and cols
    unique_rows, unique_cols = unique_coords.T
    return (aggregated_values, (unique_rows, unique_cols))


# Create session
session = cm.Session(
    name="main",
    data_path="data",
    data_path_default=CM_BASEPATH + "/data/default",
)

# Load scenarios
# ==============

SCENARIOS = [
    "BASELINE",
    "SCN_CORE",
    "SCN_MIN_LEY",
    "SCN_SNG",
    "SCN_ORG",
]

session.add_scenario(
    "BASELINE", years=[2020], pars="all", scenario_workbooks="default_fix"
)

session.add_scenario("SCN_CORE", years=[2020], pars="all", scenario_workbooks="base")

session.add_scenario(
    "SCN_MIN_LEY", years=[2020], pars="all", scenario_workbooks=["base", "scn-min-ley"]
)

session.add_scenario("SCN_SNG", years=[2020], pars="all", scenario_workbooks="base")
session.add_scenario("SCN_ORG", years=[2020], pars="all", scenario_workbooks="base")


def improve_numerics(self):
    from CIBUSmod.optimisation.feed_dist import IndexedMatrix

    for name, C in self.constraints.items():
        M = [obj for obj in C["pars"].values() if isinstance(obj, IndexedMatrix)]
        assert len(M) == 1, "Expected one and only one IndexedMatrix"
        M = M[0]
        max_val = M.M.max()
        if max_val == 1:
            continue
        for name, obj in C["pars"].items():
            if name == "tol":
                continue
            if isinstance(obj, IndexedMatrix):
                obj.M = obj.M / max_val
            else:
                obj[:] = obj / max_val

    print("Completed rescaling of matrices.")


for scn in SCENARIOS:
    print("/"*80)
    print("/"*80)
    print("/"*80)
    print("/"*80)
    print("/"*80)
    print("")
    print(scn)
    print("")
    print("\\"*80)
    print("\\"*80)
    print("\\"*80)
    print("\\"*80)
    print("\\"*80)

    retrievers = {
        "Regions": cm.ParameterRetriever("Regions"),
        "DemandAndConversions": cm.ParameterRetriever("DemandAndConversions"),
        "CropProduction": cm.ParameterRetriever("CropProduction"),
        "FeedMgmt": cm.ParameterRetriever("FeedMgmt"),
        "GeoDistributor": cm.ParameterRetriever("GeoDistributor"),
    }

    cm.ParameterRetriever.update_all_parameter_values(**session[scn], year=2020)

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
    herds = cm.make_herds(
        regions,
        sub_systems={
            "sheep": ["autumn lamb", "spring lamb", "winter lamb", "other sheep"]
        },
    )

    # Instantiate feed management
    feed_mgmt = cm.FeedMgmt(
        herds=herds,
        par=retrievers["FeedMgmt"],
        type="FeedDist"
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

    # Instantiate WasteAndCircularity
    waste = cm.WasteAndCircularity(
        demand=demand,
        crops=crops,
        herds=herds,
        par=cm.ParameterRetriever("WasteAndCircularity"),
    )

    # Instantiate by-product management
    byprod_mgmt = cm.ByProductMgmt(
        demand=demand, herds=herds, par=cm.ParameterRetriever("ByProductMgmt")
    )

    # Instantiate manure management
    manure_mgmt = cm.ManureMgmt(
        herds=herds,
        feed_mgmt=feed_mgmt,
        par=cm.ParameterRetriever("ManureMgmt"),
        settings={"NPK_excretion_from_balance": True},
    )

    # Instantiate crop residue managment
    crop_residue_mgmt = cm.CropResidueMgmt(
        demand=demand,
        crops=crops,
        herds=herds,
        par=cm.ParameterRetriever("CropResidueMgmt"),
    )

    # Instantiate plant nutrient management
    plant_nutrient_mgmt = cm.PlantNutrientMgmt(
        demand=demand,
        regions=regions,
        crops=crops,
        waste=waste,
        herds=herds,
        par=cm.ParameterRetriever("PlantNutrientMgmt"),
    )

    # Instatiate machinery and energy management
    machinery_and_energy_mgmt = cm.MachineryAndEnergyMgmt(
        regions=regions,
        crops=crops,
        waste=waste,
        herds=herds,
        par=cm.ParameterRetriever("MachineryAndEnergyMgmt"),
    )

    # Instatiate inputs management
    inputs = cm.InputsMgmt(
        demand=demand,
        crops=crops,
        waste=waste,
        herds=herds,
        par=cm.ParameterRetriever("InputsMgmt"),
    )

    def mgmt_calculate():
        # Calculate feeds
        feed_mgmt.calculate()
        # Calculate byprod
        byprod_mgmt.calculate()
        # Calculate manure
        manure_mgmt.calculate()
        # Calculate harvest of crop residues
        crop_residue_mgmt.calculate()
        # Calculate treatment of wastes and other feedstocks
        waste.calculate()
        # Calculate plant nutrient management
        plant_nutrient_mgmt.calculate()
        # Calculate energy requirements
        machinery_and_energy_mgmt.calculate()
        # Calculate inputs supply chain emissions
        inputs.calculate()

    cm.ParameterRetriever.update_all_parameter_values()
    cm.ParameterRetriever.update_relation_tables()

    cm.ParameterRetriever.update_all_parameter_values(**session[scn], year=2020)

    regions.calculate()
    demand.calculate()
    crops.calculate()
    for h in herds:
        h.calculate()

    cons = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 14]

    self.get_x0()

    sng_areas = self.x0["crp"].loc[
        [
            "Semi-natural meadows",
            "Semi-natural pastures",
            "Semi-natural pastures, thin soils",
            "Semi-natural pastures, wooded",
        ]
    ]
    fallow_areas = self.x0["crp"].loc[["Fallow"]]

    C8_params = {
        "C8_crp" : [sng_areas,  fallow_areas],
        "C8_rel" : ["<=",       "=="],
        "C8_tol" : [None,       1e-4],
    }

    if scn == "BASELINE":
        cons.remove(8)
    elif scn == "SCN_SNG":
        tol = 1e-3
        C8_params["C8_crp"][0] = sng_areas * (1 - tol)
        C8_params["C8_rel"][0] = ">="

    self.make(cons, verbose=True, **C8_params)

    def make_CX_organic_cattle(tol=0.001):
        cr_x0_org = self.x0["ani"][["cattle"]].xs("organic", level="prod_system")
        cr_x0_con = self.x0["ani"][["cattle"]].xs("conventional", level="prod_system")
        both_zero = cr_x0_org[cr_x0_org == 0].index.intersection(
            cr_x0_con[cr_x0_con == 0].index
        )
        cr_x0_org = cr_x0_org.reindex(cr_x0_org.index.difference(both_zero))

        shares_df = (
            (cr_x0_org / cr_x0_con.reindex(cr_x0_org.index, fill_value=0))
            .replace({np.inf: np.nan, -np.inf: np.nan})
            .fillna(1)
            .to_frame(name="share")
        )
        shares_df *= 1 - tol
        shares_df = shares_df.reset_index()

        col_idx = self.x_idx["ani"]
        row_idx = cr_x0_org.index

        row_idx_df = row_idx.to_frame(index=False).reset_index(names="row_i")
        col_idx_df = col_idx.to_frame(index=False).reset_index(names="col_i")

        merged = row_idx_df.merge(col_idx_df, on=row_idx.names).merge(
            shares_df, on=["species", "breed", "region"]
        )
        merged["values"] = np.where(
            merged["prod_system"] == "organic", 1 - merged["share"], -merged["share"]
        )

        n_rows = len(row_idx)
        n_cols = len(col_idx)

        row_i = merged["row_i"].to_numpy()
        col_i = merged["col_i"].to_numpy()
        values = merged["values"].to_numpy()

        M = scipy.sparse.hstack(
            [
                scipy.sparse.coo_array(
                    (values, (row_i, col_i)), shape=(n_rows, n_cols)
                ).tocsc(),
                scipy.sparse.csc_array((n_rows, len(self.x_idx["crp"]))),
                scipy.sparse.csc_array((n_rows, len(self.x_idx["fds"]))),
            ],
            format="csc",
        )

        IM = IndexedMatrix(M, row_idx=row_idx, col_idx={})

        self.constraints["CX: Share of organic cattle"] = {
            "left": lambda x, A: A.M @ x,
            "right": lambda A: 0,
            "rel": ">=",
            "pars": {"A": IM},
        }
        print(
            "Added constraint ensuring a maintained share of organic cattle production."
        )

    if "_ORG" in scn:
        make_CX_organic_cattle()
        if 7 in cons:
            self.make_C7()

    improve_numerics(optproblem)

    # =========================================================================== #
    # --------------- SOLVE, STORE, AND STOP IF ON BASELINE SCENARIO ------------ #
    # =========================================================================== #

    if scn == "BASELINE":
        self.solve(
            apply_solution=True,
            verbose=True,
            solver_settings=[
                {
                    "solver": "GUROBI",
                    "reoptimize": True,
                    "verbose": True,
                }
            ],
        )
        mgmt_calculate()
        session.store(scn, 2020, demand, regions, crops, herds, optproblem, waste)

        continue

    # =========================================================================== #
    # ---------------------------- OTHERWISE, CONTINUE -------------------------- #
    # =========================================================================== #

    ADDED_PEAS = "Peas (add)"
    ADDED_WHEAT = "Wheat (add)"

    PROTEIN_CONTENTS = {
        ADDED_PEAS: 220,
        ADDED_WHEAT: 67.15,
        "meat": 155.5,
        "milk": 35.0,
    }

    # Convert to thousands of prot. / kg, instead of straight
    for k in PROTEIN_CONTENTS.keys():
        PROTEIN_CONTENTS[k] /= 1e3

    def make_protein_mask_ani():
        RELEVANT_ANIMAL_PRODUCTS = ["meat", "milk"]

        # Get row index from animal product demand vector (ps,sp,ap)
        row_idx = pd.MultiIndex.from_tuples(
            [
                ("conventional", "cattle", "meat"),
                ("conventional", "cattle", "milk"),
                ("organic", "cattle", "meat"),
                ("organic", "cattle", "milk"),
            ],
            names=["prod_system", "species", "animal_prod"],
        )

        # Get col index from animal herds (sp,br,ss,ps,re)
        col_idx = self.x_idx_short["ani"]

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
                if (ops, sp, ap) not in row_idx:
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
                col_nr.extend(
                    [col_idx.get_loc((sp, br, ps, ss, re)) for re in res.index]
                )
                row_nr.extend(np.zeros(len(res)))

        # Aggregate data_coords_pair to ensure that any overlapping values are summed rather than replace each other
        val, (row_nr, col_nr) = aggregate_data_coords_pair(val, row_nr, col_nr)

        # Create Compressed Sparse Column matrix
        return scipy.sparse.coo_array(
            (val, (row_nr, col_nr)), shape=(1, len(col_idx))
        ).tocsc()

    def make_protein_mask_crp():
        val_df = crops.data_attr.get("harvest").loc[["Wheat (add)"]].reindex(
            self.x_idx_short["crp"]
        ).fillna(0) * [PROTEIN_CONTENTS[ADDED_WHEAT]] + crops.data_attr.get(
            "harvest"
        ).loc[["Peas (add)"]].reindex(self.x_idx_short["crp"]).fillna(0) * [
            PROTEIN_CONTENTS[ADDED_PEAS]
        ]

        return scipy.sparse.csc_array(np.atleast_2d(val_df.values))

    def make_protein_mask():
        A_ani = make_protein_mask_ani()
        A_crp = make_protein_mask_crp()
        A_fds = scipy.sparse.csc_matrix((1, len(self.x_idx_short["fds"])))

        return scipy.sparse.hstack([A_ani, A_crp, A_fds], format="csc")

    prot_mask = make_protein_mask()

    def protein_mask_as_opt_goal():
        n = (
            len(self.x_idx_short["ani"])
            + len(self.x_idx_short["crp"])
            + len(self.x_idx_short["fds"])
        )
        x = cvxpy.Variable(n, nonneg=True)

        M = prot_mask
        objective = cvxpy.Maximize(cvxpy.sum(M @ x))

        # Append constraints
        constraints = [
            make_cvxpy_constraint(cons, x) for cons in self.constraints.values()
        ]

        # Define problem
        self.problem = cvxpy.Problem(objective=objective, constraints=constraints)

    protein_mask_as_opt_goal()
    self.solve(
        apply_solution=False,
        verbose=True,
        solver_settings=[
            {
                "solver": "GUROBI",
                "reoptimize": True,
                "verbose": True,
                # "GURO_PAR_DUMP": 1,
                # Custom params
                "BarConvTol": 1e-8,
                # Sometimes setting Aggregate=0 can improve the model numerics
                # "Aggregate": 0,
                "NumericFocus": 3,
                # Useful for recognizing infeasibility or unboundedness, but a bit slower than the default algorithm.
                # values: -1 auto, 0 off, 1 force on.
                "BarHomogeneous": 1,
                # Gurobi has three different heuristic algorithms to find scaling factors. Higher values for the ScaleFlag uses more aggressive heuristics to improve the constraint matrix numerics for the scaled model.
                "ScaleFlag": 2,
                # All constraints must be satisfied to a tolerance of FeasibilityTol.
                # default 1e-6
                "FeasibilityTol": 1e-3,
            }
        ],
    )

    print(f"Max protein calculated as:\n{self.problem.value:e}")
    max_protein_amount = self.problem.value

    def protein_map_as_cons(max_protein_amount):
        if max_protein_amount is None:
            raise Exception("Could not get the optimal value from the problem")

        b = max_protein_amount

        return {
            "left": lambda x, M, b: M @ x - b,
            "right": lambda M, b: 0,
            "rel": ">=",
            "pars": {"M": prot_mask, "b": b},
        }

    self.constraints["CX: Protein"] = protein_map_as_cons(0.975 * max_protein_amount)
    # Overwrite old problem with standard optimization objective,
    # but with the protein constraint added
    self.problem = self.define_cvx_problem()

    self.solve(
        apply_solution=False,
        verbose=True,
        solver_settings=[
            {
                "solver": "GUROBI",
                "reoptimize": True,
                "verbose": True,

                "BarConvTol": 1e-3,
            }
        ],
    )
    self.apply_solution()
    mgmt_calculate()

    session.store(scn, 2020, demand, regions, crops, herds, optproblem, waste)
