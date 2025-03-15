import csv
import os
import sys
import time

CM_BASEPATH = "../cibusmod"
sys.path.insert(0, os.path.join(os.getcwd(), CM_BASEPATH))
import CIBUSmod as cm

# Create session
session = cm.Session(
    name="time-eval",
    data_path="data",
    data_path_default=CM_BASEPATH + "/data/default",
)

session.add_scenario(
    name="base",
    years=[2020],
    scenario_workbooks="default_fix"
)

retrievers = {
    "Regions": cm.ParameterRetriever("Regions"),
    "DemandAndConversions": cm.ParameterRetriever("DemandAndConversions"),
    "CropProduction": cm.ParameterRetriever("CropProduction"),
    "FeedMgmt": cm.ParameterRetriever("FeedMgmt"),
    "GeoDistributor": cm.ParameterRetriever("GeoDistributor"),
}

cm.ParameterRetriever.update_all_parameter_values(**session["base"], year=2020)

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
    sub_systems={"sheep": ["autumn lamb", "spring lamb", "winter lamb", "other sheep"]},
)

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

regions.calculate(verbose=True)
demand.calculate(verbose=True)
crops.calculate(verbose=True)
for h in herds:
    h.calculate(verbose=True)

n_iterations = 30
self.make(use_cons=[1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14], verbose=True)

FNAME = "solve-timings.feeddist.csv"
if not os.path.exists(FNAME):
    with open(FNAME, "w+", newline="") as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerow(["Iteration", "Duration"])

for i in range(n_iterations):
    self.problem = self.get_cvx_problem()
    success = False
    print(f"Iteration {i + 1} / {n_iterations}...")
    t0 = time.time()
    self.solve(
        {
            "solver": "GUROBI",
            "reoptimize": True,
            "verbose": True,
        }
    )
    duration = time.time() - t0

    with open(FNAME, "a", newline="") as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerow([i, duration])
