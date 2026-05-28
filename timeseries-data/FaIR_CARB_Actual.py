#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 14:36:32 2026

@author: ryoung
"""

"""
Created on Thu Apr 23 15:36:39 2026

@author: ryoung
"""

"""
FaIR analysis: temperature implications of offsetting under CO2e equivalence

assumption:
- offset-justified emissions and credited project fluxes are assumed to be
  approximately equal in cumulative co2e space (i.e. the offset transaction balances)

purpose:
- test whether equal co2e perturbations produce equal temperature outcomes

scenarios:
- baseline: ssp2-4.5
- emissions_only: ssp245 + offset-justified emissions (demand side)
- projects_only: ssp245 + credited project fluxes (supply side)
- net_program: ssp245 + offset-justified emissions + project fluxes

interpretation:
- comparing emissions_only and projects_only isolates whether physically distinct
  but co2e-equivalent perturbations have different temperature impacts
- net_program shows the temperature response of the offset system as a whole

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 
from fair import FAIR
from fair.interface import fill, initialise
from fair.io import read_properties


# ----------------------------
# Inputs
# ----------------------------

PROJECTS_FILE = "output-data/ca_project_fluxes.csv"
EMISSIONS_FILE = "output-data/ca_justified_emissions_fluxes.csv"

BASELINE_SCENARIO = "ssp245"


# ----------------------------
# load data
# ----------------------------

df_p = pd.read_csv(PROJECTS_FILE)
df_p.columns = df_p.columns.str.lower()
df_p = df_p[["year", "co2_flux", "ch4_flux", "cfc-12_flux"]].sort_values("year").fillna(0)

df_e = pd.read_csv(EMISSIONS_FILE)
df_e.columns = df_e.columns.str.lower()

emis_cols = ["year","co2_justified","ch4_justified"]
if "n2o_flux" in df_e.columns:
    emis_cols.append("n2o_justified")

df_e = df_e[emis_cols].sort_values("year").fillna(0)


# ----------------------------
# set up fair
# ----------------------------

f = FAIR()
f.define_time(1750, 2200, 1)

f.define_scenarios([BASELINE_SCENARIO,"emissions_only", "projects_only", "net_program"])

f.define_configs(["central"])

all_species, all_props = read_properties()

species = ["CO2 FFI", "CO2 AFOLU", "CH4", "N2O", "CFC-12", "CO2"]
properties = {s: all_props[s] for s in species}

properties["CO2 FFI"]["input_mode"] = "emissions"
properties["CO2 FFI"]["greenhouse_gas"] = False

properties["CO2 AFOLU"]["input_mode"] = "emissions"
properties["CO2 AFOLU"]["greenhouse_gas"] = False

properties["CH4"]["input_mode"] = "emissions"
properties["CH4"]["greenhouse_gas"] = True

properties["N2O"]["input_mode"] = "emissions"
properties["N2O"]["greenhouse_gas"] = True

properties["CFC-12"]["input_mode"] = "emissions"
properties["CFC-12"]["greenhouse_gas"] = True

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

for sp in ["CO2 FFI", "CO2 AFOLU", "CH4", "N2O", "CFC-12"]:

    base = f.emissions.sel(scenario=BASELINE_SCENARIO, config="central", specie=sp).values

    for scen in ["emissions_only","projects_only","net_program"]:
        fill(f.emissions, base, scenario=scen, config="central", specie=sp)


# ----------------------------
# apply project perturbations
# ----------------------------

timepoints = np.asarray(f.timepoints, dtype=float)

for _, row in df_p.iterrows():

    year = float(row["year"])
    i = np.argmin(np.abs(timepoints - year))
    yr = timepoints[i]

    co2_p = row["co2_flux"] / 1e9           # Conversion from ton to gigaton
    ch4_p = row["ch4_flux"] / 1e6           # Conversion from ton to megaton
    cfc12_p = row["cfc-12_flux"] / 1e3      # Conversion from ton to kilaton

    for scen in ["projects_only", "net_program"]:

        f.emissions.loc[dict(timepoints=yr, scenario=scen, config="central", specie="CO2 FFI")].values += co2_p

        f.emissions.loc[dict(timepoints=yr, scenario=scen, config="central", specie="CH4")].values += ch4_p

        f.emissions.loc[dict(timepoints=yr, scenario=scen, config="central", specie="CFC-12")].values += cfc12_p


# ----------------------------
# apply offset-justified emissions
# ----------------------------

for _, row in df_e.iterrows():

    year = float(row["year"])
    i = np.argmin(np.abs(timepoints - year))
    yr = timepoints[i]

    co2_e = row["co2_justified"] / 1e9
    ch4_e = row["ch4_justified"] / 1e6

    n2o_e = 0
    if "n2o_flux" in df_e.columns:
        n2o_e = row["n2o_justified"] / 1e6

    for scen in ["emissions_only", "net_program"]:

        f.emissions.loc[dict(timepoints=yr, scenario=scen, config="central", specie="CO2 FFI")].values += co2_e

        f.emissions.loc[dict(timepoints=yr, scenario=scen, config="central", specie="CH4")].values += ch4_e

        f.emissions.loc[dict(timepoints=yr, scenario=scen, config="central", specie="N2O")].values += n2o_e


# ----------------------------
# initialise + run
# ----------------------------

initialise(f.concentration, 278.0, specie="CO2")
initialise(f.concentration, 730.0, specie="CH4")
initialise(f.concentration, 270.0, specie="N2O")
initialise(f.concentration, 0.0, specie="CFC-12")

initialise(f.forcing, 0.0)
initialise(f.temperature, 0.0)
initialise(f.cumulative_emissions, 0.0)
initialise(f.airborne_emissions, 0.0)
initialise(f.ocean_heat_content_change, 0.0)

f.run()

#%%
# ----------------------------
# plot temperature trajectories
# ----------------------------

years = np.asarray(f.timebounds, dtype=float)
baseline = f.temperature.loc[dict(scenario=BASELINE_SCENARIO, config="central", layer=0)]
emissions_only = f.temperature.loc[dict(scenario="emissions_only", config="central", layer=0)]
projects_only = f.temperature.loc[dict(scenario="projects_only", config="central", layer=0)]
net_program = f.temperature.loc[dict(scenario="net_program", config="central", layer=0)]


plt.figure(figsize=(7,4))

plt.plot(years, emissions_only - baseline, label="Offset-justified emissions")
plt.plot(years, projects_only - baseline, label="Project fluxes")
plt.plot(years, net_program - baseline, label="Net offset program", linewidth=2)

plt.axhline(0, ls="--", color="k")
plt.xlim(2000,2200)
plt.xlabel("Year")
plt.ylabel("ΔT relative to baseline (K)")
plt.title("Temperature impacts relative to SSP2-4.5")
plt.legend(fontsize=9)
plt.tight_layout()
plt.savefig("output-figs/fair_offset_program_actual.png", dpi=300, bbox_inches="tight")
plt.show()

print("done: ran offset program perturbation analysis")