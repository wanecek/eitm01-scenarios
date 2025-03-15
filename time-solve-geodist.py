import csv
import os
import sys
import time

CM_BASEPATH = "../cibusmod"
sys.path.insert(0, os.path.join(os.getcwd(), CM_BASEPATH))
import CIBUSmod as cm

# Create session
session = cm.Session(
    name="main",
    data_path="data",
    data_path_default=CM_BASEPATH + "/data/default",
)

session.add_scenario(name="none", years=[2020])
# Instatiate Regions
regions = cm.Regions(par=cm.ParameterRetriever("Regions"))

# Instantiate DemandAndConversions
demand = cm.DemandAndConversions(par=cm.ParameterRetriever("DemandAndConversions"))

# Instantiate CropProduction
crops = cm.CropProduction(
    par=cm.ParameterRetriever("CropProduction"),
    index=regions.data_attr.get("x0_crops").index,
)

# Instantiate AnimalHerds
# Each AnimalHerd object is stored in an indexed pandas.Series
herds = cm.make_herds(regions)

# Instantiate feed management
feed_mgmt = cm.FeedMgmt(herds=herds, par=cm.ParameterRetriever("FeedMgmt"))

# Instantiate feed distributor
self = cm.GeoDistributor(
    regions=regions,
    demand=demand,
    crops=crops,
    herds=herds,
    feed_mgmt=feed_mgmt,
    par=cm.ParameterRetriever("GeoDistributor"),
)

cm.ParameterRetriever.update_all_parameter_values()
cm.ParameterRetriever.update_relation_tables()
regions.calculate(verbose=True)
demand.calculate(verbose=True)
crops.calculate(verbose=True)
for h in herds:
    h.calculate(verbose=True)

feed_mgmt.calculate(verbose=True)

self.make(use_cons=[], verbose=True)

cons_to_test = [1, 2, 3, 4, 5, 6, 7]
n_iterations = 30

FNAME = "solve-timings.geodist.csv"
if not os.path.exists(FNAME):
    with open(FNAME, "w+", newline="") as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerow(["Constraint", "Iteration", "Duration"])

for i in range(n_iterations):
    self.make_cvx_problem()
    success = False
    print(f"Iteration {i + 1} / {n_iterations}...")
    t0 = time.time()
    self.solve({
        "solver": "GUROBI",
        "reoptimize": True,
        "verbose": True,
    })
    duration = time.time() - t0

    with open(FNAME, "a", newline="") as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerow([i, duration])
