#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 15:36:39 2026

@author: ryoung
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FaIR analysis: counterfactual vs actual vs claimed

inputs:
- timeseries_emissions_fluxes.csv
- timeseries_project_fluxes.csv

scenarios:
- counterfactual: emissions only (no projects)
- actual: emissions + project reductions
- claimed: CH4 reductions treated as CO2 via GWP100
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from fair import FAIR
from fair.interface import fill, initialise
from fair.io import read_properties


# ----------------------------
# user inputs
# ----------------------------

EMISSIONS_FILE = "timeseries_emission_fluxes.csv"
PROJECTS_FILE = "timeseries_project_fluxes.csv"

BASELINE_SCENARIO = "ssp245"
# year-dependent GWP100 for CH4 (matches CARB data)
def gwp_ch4(year):
    return 21 if year <= 2020 else 25


# ----------------------------
# load data
# ----------------------------

emissions = pd.read_csv(EMISSIONS_FILE)
projects = pd.read_csv(PROJECTS_FILE)

emissions.columns = emissions.columns.str.lower()
projects.columns = projects.columns.str.lower()

df = emissions.merge(
    projects[["year", "co2_flux", "ch4_flux"]],
    on="year",
    how="outer",
    suffixes=("_e", "_p")
).sort_values("year").fillna(0)


# ----------------------------
# set up fair
# ----------------------------

f = FAIR()
f.define_time(1750, 2200, 1)
f.define_scenarios([
    BASELINE_SCENARIO,
    "counterfactual",
    "actual",
    "claimed",
    "counterfactual_ch4",
    "actual_ch4",
    "claimed_ch4",
])
f.define_configs(["central"])

all_species, all_props = read_properties()

species = ["CO2 FFI", "CO2 AFOLU", "CH4", "N2O", "CO2"]
properties = {s: all_props[s] for s in species}

properties["CO2 FFI"]["input_mode"] = "emissions"
properties["CO2 FFI"]["greenhouse_gas"] = False

properties["CO2 AFOLU"]["input_mode"] = "emissions"
properties["CO2 AFOLU"]["greenhouse_gas"] = False

properties["CH4"]["input_mode"] = "emissions"
properties["CH4"]["greenhouse_gas"] = True

properties["N2O"]["input_mode"] = "emissions"
properties["N2O"]["greenhouse_gas"] = True

properties["CO2"]["input_mode"] = "calculated"
properties["CO2"]["greenhouse_gas"] = True

f.define_species(species, properties)
f.allocate()
f.fill_species_configs()

fill(f.climate_configs["ocean_heat_capacity"], [8.0, 14.0, 100.0], config="central")
fill(f.climate_configs["ocean_heat_transfer"], [1.1, 1.6, 0.9], config="central")
fill(f.climate_configs["deep_ocean_efficacy"], 1.1, config="central")


# ----------------------------
# load baseline emissions
# ----------------------------

backup = list(f.scenarios)
f.scenarios = [BASELINE_SCENARIO]
f.fill_from_rcmip()
f.scenarios = backup

for sp in ["CO2 FFI", "CO2 AFOLU", "CH4", "N2O"]:
    base = f.emissions.sel(
        scenario=BASELINE_SCENARIO,
        config="central",
        specie=sp
    ).values

    for scen in ["counterfactual", "actual", "claimed",
             "counterfactual_ch4", "actual_ch4", "claimed_ch4"]:
        fill(f.emissions, base, scenario=scen, config="central", specie=sp)


# ----------------------------
# apply perturbations
# ----------------------------

timepoints = np.asarray(f.timepoints, dtype=float)

for _, row in df.iterrows():

    year = float(row["year"])
    i = np.argmin(np.abs(timepoints - year))
    yr = timepoints[i]

    # convert units
    co2_e = row["co2_flux_e"] / 1e9
    ch4_e = row["ch4_flux_e"] / 1e6

    co2_p = row["co2_flux_p"] / 1e9
    ch4_p = row["ch4_flux_p"] / 1e6

    # counterfactual
    f.emissions.loc[dict(timepoints=yr, scenario="counterfactual", config="central", specie="CO2 FFI")].values += co2_e
    f.emissions.loc[dict(timepoints=yr, scenario="counterfactual", config="central", specie="CH4")].values += ch4_e

    # actual
    f.emissions.loc[dict(timepoints=yr, scenario="actual", config="central", specie="CO2 FFI")].values += co2_e + co2_p
    f.emissions.loc[dict(timepoints=yr, scenario="actual", config="central", specie="CH4")].values += ch4_e + ch4_p

    # claimed
    f.emissions.loc[dict(timepoints=yr, scenario="claimed", config="central", specie="CO2 FFI")].values += co2_e + co2_p + ch4_p * gwp_ch4(year) / 1000
    f.emissions.loc[dict(timepoints=yr, scenario="claimed", config="central", specie="CH4")].values += ch4_e
    
    # ----------------------------
    # ch4-only scenarios
    # ----------------------------
    
    # counterfactual_ch4
    f.emissions.loc[dict(timepoints=yr, scenario="counterfactual_ch4", config="central", specie="CH4")].values += ch4_e
    
    # actual_ch4
    f.emissions.loc[dict(timepoints=yr, scenario="actual_ch4", config="central", specie="CH4")].values += ch4_e + ch4_p
    
    # claimed_ch4
    f.emissions.loc[dict(timepoints=yr, scenario="claimed_ch4", config="central", specie="CH4")].values += ch4_e
    f.emissions.loc[dict(timepoints=yr, scenario="claimed_ch4", config="central", specie="CO2 FFI")].values += ch4_p * gwp_ch4(year) / 1000

# ----------------------------
# initialise + run
# ----------------------------

initialise(f.concentration, 278.0, specie="CO2")
initialise(f.concentration, 730.0, specie="CH4")
initialise(f.concentration, 270.0, specie="N2O")

initialise(f.forcing, 0.0)
initialise(f.temperature, 0.0)
initialise(f.cumulative_emissions, 0.0)
initialise(f.airborne_emissions, 0.0)
initialise(f.ocean_heat_content_change, 0.0)

f.run()

#%% Plot emissions

years_e = np.asarray(f.timepoints, dtype=float)

baseline_co2 = f.emissions.loc[dict(scenario=BASELINE_SCENARIO, config="central", specie="CO2 FFI")]
baseline_ch4 = f.emissions.loc[dict(scenario=BASELINE_SCENARIO, config="central", specie="CH4")]

counterfactual_co2 = f.emissions.loc[dict(scenario="counterfactual", config="central", specie="CO2 FFI")]
actual_co2 = f.emissions.loc[dict(scenario="actual", config="central", specie="CO2 FFI")]
claimed_co2 = f.emissions.loc[dict(scenario="claimed", config="central", specie="CO2 FFI")]

counterfactual_ch4 = f.emissions.loc[dict(scenario="counterfactual", config="central", specie="CH4")]
actual_ch4 = f.emissions.loc[dict(scenario="actual", config="central", specie="CH4")]
claimed_ch4 = f.emissions.loc[dict(scenario="claimed", config="central", specie="CH4")]

# co2
plt.figure(figsize=(7, 4))
plt.plot(years_e, counterfactual_co2 - baseline_co2, label="Counterfactual")
plt.plot(years_e, actual_co2 - baseline_co2, label="Actual")
plt.plot(years_e, claimed_co2 - baseline_co2, label="Claimed")
plt.axhline(0, linestyle="--", linewidth=1)
plt.xlabel("Year")
plt.ylabel("CO2 emissions perturbation (GtCO2/yr)")
plt.title("CO2 emissions relative to baseline")
plt.legend()
plt.tight_layout()
plt.show()

# ch4
plt.figure(figsize=(7, 4))
plt.plot(years_e, counterfactual_ch4 - baseline_ch4, label="Counterfactual")
plt.plot(years_e, actual_ch4 - baseline_ch4, label="Actual")
plt.plot(years_e, claimed_ch4 - baseline_ch4, label="Claimed")
plt.axhline(0, linestyle="--", linewidth=1)
plt.xlabel("Year")
plt.ylabel("CH4 emissions perturbation (MtCH4/yr)")
plt.title("CH4 emissions relative to baseline")
plt.legend()
plt.tight_layout()
plt.show()


#%% plot temperature trajectories

years = np.asarray(f.timebounds, dtype=float)

baseline = f.temperature.loc[dict(scenario=BASELINE_SCENARIO, config="central", layer=0)]
counterfactual = f.temperature.loc[dict(scenario="counterfactual", config="central", layer=0)]
actual = f.temperature.loc[dict(scenario="actual", config="central", layer=0)]
claimed = f.temperature.loc[dict(scenario="claimed", config="central", layer=0)]

# absolute
plt.figure(figsize=(7, 4))
plt.plot(years, baseline, "--", label="Baseline (SSP2-4.5)")
plt.plot(years, counterfactual, label="Counterfactual")
plt.plot(years, actual, label="Actual")
plt.plot(years, claimed, label="Claimed")
plt.xlabel("Year")
plt.ylabel("Temperature anomaly (K)")
plt.title("Temperature trajectories")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 4))
# total effects (solid)
plt.plot(years, actual - counterfactual, label="Actual (total)")
plt.plot(years, claimed - counterfactual, label="Claimed (total)")
# ch4-only effects (dashed)
plt.plot(years,
         f.temperature.loc[dict(scenario="actual_ch4", config="central", layer=0)]
         - f.temperature.loc[dict(scenario="counterfactual_ch4", config="central", layer=0)],
         "--", label="Actual (CH4 only)")

plt.plot(years,
         f.temperature.loc[dict(scenario="claimed_ch4", config="central", layer=0)]
         - f.temperature.loc[dict(scenario="counterfactual_ch4", config="central", layer=0)],
         "--", label="Claimed (CH4 only)")

plt.axhline(0, linestyle="--", linewidth=1,color='k')
plt.ylim(-0.0001, 0.00001)
plt.xlim(2000, 2200)
plt.xlabel("Year")
plt.ylabel("ΔT relative to counterfactual (K)")
plt.title("Temperature impact")
plt.legend()
plt.tight_layout()
plt.show()
