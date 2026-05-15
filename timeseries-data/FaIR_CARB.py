#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 15:36:39 2026

@author: ryoung
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Thu Apr 23 15:36:39 2026

@author: ryoung
"""

"""
FaIR analysis: physical project mitigation vs claimed co2e mitigation

inputs:
- timeseries_project_fluxes.csv

scenarios:
- baseline: ssp2-4.5 only
- actual: baseline + physical project co2, ch4, and cfc-12 fluxes
- claimed: baseline + project co2 fluxes + ch4 and cfc-12 project fluxes converted to co2e
- actual_ch4: baseline + physical ch4 project fluxes only
- claimed_ch4: baseline + ch4 project fluxes converted to co2e only
- actual cfc-12: baseline + physical cfc-12 project fluxes only
- claimed cfc-12: baseline + cfc-12 project fluxes converted to co2e only

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

PROJECTS_FILE = "output-data/ca_project_fluxes.csv"

BASELINE_SCENARIO = "ssp245"

# year-dependent gwp100 for ch4
def gwp_ch4(year):
    return 21 if year <= 2020 else 25

# gwp100 for cfc-12
gwp_cfc12 = 10900


# ----------------------------
# load data
# ----------------------------

df = pd.read_csv(PROJECTS_FILE)
df.columns = df.columns.str.lower()
df = df[["year", "co2_flux", "ch4_flux", "cfc-12_flux"]].sort_values("year").fillna(0)


# ----------------------------
# set up fair
# ----------------------------

f = FAIR()
f.define_time(1750, 2200, 1)
f.define_scenarios([
    BASELINE_SCENARIO,
    "actual",
    "claimed",
    "actual_ch4",
    "claimed_ch4",
    "actual_cfc12",
    "claimed_cfc12"
])
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
    base = f.emissions.sel(
        scenario=BASELINE_SCENARIO,
        config="central",
        specie=sp
    ).values

    for scen in ["actual", "claimed", "actual_ch4", "claimed_ch4", "actual_cfc12", "claimed_cfc12"]:
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
    co2_p = row["co2_flux"] / 1e9       # Conversion from ton to gigaton
    ch4_p = row["ch4_flux"] / 1e6       # Conversion from ton to megaton
    cfc12_p = row["cfc-12_flux"] / 1e3  # Conversion from ton to kilaton

    # actual: all project fluxes enter as physical gases
    f.emissions.loc[dict(timepoints=yr, scenario="actual", config="central", specie="CO2 FFI")].values += co2_p
    f.emissions.loc[dict(timepoints=yr, scenario="actual", config="central", specie="CH4")].values += ch4_p
    f.emissions.loc[dict(timepoints=yr, scenario="actual", config="central", specie="CFC-12")].values += cfc12_p

    # claimed: co2 project fluxes + ch4 and cfc12 project fluxes represented as co2-equivalent
    f.emissions.loc[dict(timepoints=yr, scenario="claimed", config="central", specie="CO2 FFI")].values += co2_p + ch4_p * gwp_ch4(year) / 1000 + cfc12_p * gwp_cfc12 / 1e6

    # actual_ch4: only physical ch4 project flux
    f.emissions.loc[dict(timepoints=yr, scenario="actual_ch4", config="central", specie="CH4")].values += ch4_p

    # claimed_ch4: only ch4 project flux represented as co2-equivalent
    f.emissions.loc[dict(timepoints=yr, scenario="claimed_ch4", config="central", specie="CO2 FFI")].values += ch4_p * gwp_ch4(year) / 1000

    # actual_cfc12: only physical cfc-12 project flux
    f.emissions.loc[dict(timepoints=yr, scenario="actual_cfc12", config="central", specie="CFC-12")].values += cfc12_p

    # claimed_cfc12: only cfc-12 flux represented as co2-equivalent
    f.emissions.loc[dict(timepoints=yr, scenario="claimed_cfc12", config="central", specie="CO2 FFI")].values += cfc12_p * gwp_cfc12 / 1e6


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


#%% plot emissions

years_e = np.asarray(f.timepoints, dtype=float)

baseline_co2 = f.emissions.loc[dict(scenario=BASELINE_SCENARIO, config="central", specie="CO2 FFI")]
baseline_ch4 = f.emissions.loc[dict(scenario=BASELINE_SCENARIO, config="central", specie="CH4")]

actual_co2 = f.emissions.loc[dict(scenario="actual", config="central", specie="CO2 FFI")]
claimed_co2 = f.emissions.loc[dict(scenario="claimed", config="central", specie="CO2 FFI")]
actual_ch4_only_co2 = f.emissions.loc[dict(scenario="actual_ch4", config="central", specie="CO2 FFI")]
claimed_ch4_only_co2 = f.emissions.loc[dict(scenario="claimed_ch4", config="central", specie="CO2 FFI")]

actual_ch4 = f.emissions.loc[dict(scenario="actual", config="central", specie="CH4")]
claimed_ch4 = f.emissions.loc[dict(scenario="claimed", config="central", specie="CH4")]
actual_ch4_only = f.emissions.loc[dict(scenario="actual_ch4", config="central", specie="CH4")]
claimed_ch4_only = f.emissions.loc[dict(scenario="claimed_ch4", config="central", specie="CH4")]

# co2
plt.figure(figsize=(7, 4))
plt.plot(years_e, actual_co2 - baseline_co2, label="Actual")
plt.plot(years_e, claimed_co2 - baseline_co2, label="Claimed")
plt.plot(years_e, actual_ch4_only_co2 - baseline_co2, "--", label="Actual (CH4 only)")
plt.plot(years_e, claimed_ch4_only_co2 - baseline_co2, "--", label="Claimed (CH4 only)")
plt.axhline(0, linestyle="--", linewidth=1, color='k')
plt.xlabel("Year")
plt.ylabel("CO2 emissions perturbation (GtCO2/yr)")
plt.title("CO2 emissions relative to baseline")
plt.legend()
plt.tight_layout()
plt.show()

# ch4
plt.figure(figsize=(7, 4))
plt.plot(years_e, actual_ch4 - baseline_ch4, label="Actual")
plt.plot(years_e, claimed_ch4 - baseline_ch4, label="Claimed")
plt.plot(years_e, actual_ch4_only - baseline_ch4, "--", label="Actual (CH4 only)")
plt.plot(years_e, claimed_ch4_only - baseline_ch4, "--", label="Claimed (CH4 only)")
plt.axhline(0, linestyle="--", linewidth=1, color='k')
plt.xlabel("Year")
plt.ylabel("CH4 emissions perturbation (MtCH4/yr)")
plt.title("CH4 emissions relative to baseline")
plt.legend()
plt.tight_layout()
plt.show()


#%% plot temperature trajectories

years = np.asarray(f.timebounds, dtype=float)

baseline = f.temperature.loc[dict(scenario=BASELINE_SCENARIO, config="central", layer=0)]
actual = f.temperature.loc[dict(scenario="actual", config="central", layer=0)]
claimed = f.temperature.loc[dict(scenario="claimed", config="central", layer=0)]
actual_ch4 = f.temperature.loc[dict(scenario="actual_ch4", config="central", layer=0)]
claimed_ch4 = f.temperature.loc[dict(scenario="claimed_ch4", config="central", layer=0)]

# plot 
plt.figure(figsize=(7, 4))
plt.plot(years, actual - baseline, label="Actual (total)")
plt.plot(years, claimed - baseline, label="Claimed (total)")
plt.plot(years, actual_ch4 - baseline, "--", label="Actual (CH4 only)")
plt.plot(years, claimed_ch4 - baseline, "--", label="Claimed (CH4 only)")
plt.axhline(0, linestyle="--", linewidth=1, color='k')
plt.ylim(-0.0001, 0.00001)
plt.xlim(2000, 2200)
plt.xlabel("Year")
plt.ylabel("ΔT relative to baseline (K)")
plt.title("Temperature impact of project fluxes")
plt.legend()
plt.tight_layout()
plt.savefig("output-figs/fair_temperature_impact.png", dpi=300, bbox_inches="tight")
plt.show()

# ----------------------------
# freya plot additions
# ----------------------------

#%% plot emissions
years_e = np.asarray(f.timepoints, dtype=float)

# extract emissions
baseline_co2   = f.emissions.loc[dict(scenario=BASELINE_SCENARIO, config="central", specie="CO2 FFI")]
baseline_ch4   = f.emissions.loc[dict(scenario=BASELINE_SCENARIO, config="central", specie="CH4")]
baseline_cfc12 = f.emissions.loc[dict(scenario=BASELINE_SCENARIO, config="central", specie="CFC-12")]

actual_co2_emis        = f.emissions.loc[dict(scenario="actual", config="central", specie="CO2 FFI")]
actual_ch4_emis        = f.emissions.loc[dict(scenario="actual", config="central", specie="CH4")]
actual_cfc12_emis      = f.emissions.loc[dict(scenario="actual", config="central", specie="CFC-12")]

actual_co2_only_emis   = f.emissions.loc[dict(scenario="actual", config="central", specie="CO2 FFI")]
actual_ch4_only_emis   = f.emissions.loc[dict(scenario="actual_ch4", config="central", specie="CH4")]
actual_cfc12_only_emis = f.emissions.loc[dict(scenario="actual_cfc12", config="central", specie="CFC-12")]

claimed_co2_only_co2   = f.emissions.loc[dict(scenario="claimed", config="central", specie="CO2 FFI")]
claimed_ch4_only_co2   = f.emissions.loc[dict(scenario="claimed_ch4", config="central", specie="CO2 FFI")]
claimed_cfc12_only_co2 = f.emissions.loc[dict(scenario="claimed_cfc12", config="central", specie="CO2 FFI")]

# --- CO2 emissions: actual vs claimed (shows "addition" of co2e from ch4 and cfc-12) ---
plt.figure(figsize=(7, 4))
plt.plot(years_e, (actual_co2_only_emis - baseline_co2) * 1e9, label="Actual (tCO2/yr)")
plt.plot(years_e, (claimed_co2_only_co2 - baseline_co2) * 1e9, "--", label="Implied CO2e (tCO2e/yr)")
plt.axhline(0, linestyle="--", linewidth=1, color='k')
plt.xlim(2000, 2030)
plt.xlabel("Year")
plt.ylabel("Emissions perturbation (tons/yr)")
plt.title("CO2: actual CO2 vs implied CO2e emissions")
plt.legend()
plt.tight_layout()
plt.show()

# --- CH4 emissions: actual vs implied CO2e ---
plt.figure(figsize=(7, 4))
plt.plot(years_e, (actual_ch4_only_emis - baseline_ch4) * 1e6,      label="Actual (tCH4/yr)")
plt.plot(years_e, (claimed_ch4_only_co2 - baseline_co2) * 1e9, "--", label="Implied CO2e (tCO2e/yr)")
plt.axhline(0, linestyle="--", linewidth=1, color='k')
plt.xlim(2000, 2030)
plt.xlabel("Year")
plt.ylabel("Emissions perturbation (tons/yr)")
plt.title("CH4: actual CH4 vs implied CO2e emissions")
plt.legend()
plt.tight_layout()
plt.show()

# --- CFC-12 emissions: actual vs implied CO2e ---
plt.figure(figsize=(7, 4))
plt.plot(years_e, (actual_cfc12_only_emis - baseline_cfc12) * 1e3,  label="Actual (tCFC-12/yr)")
plt.plot(years_e, (claimed_cfc12_only_co2 - baseline_co2) * 1e9, "--", label="Implied CO2e (tCO2e/yr)")
plt.axhline(0, linestyle="--", linewidth=1, color='k')
plt.xlim(2000, 2030)
plt.yscale("symlog")
plt.xlabel("Year")
plt.ylabel("Emissions perturbation (tons/yr)")
plt.title("CFC-12: actual CFC-12 vs implied CO2e emissions")
plt.legend()
plt.tight_layout()
plt.show()

#%% plot temperature trajectories
years = np.asarray(f.timebounds, dtype=float)

baseline = f.temperature.loc[dict(scenario=BASELINE_SCENARIO, config="central", layer=0)]
actual = f.temperature.loc[dict(scenario="actual", config="central", layer=0)]
claimed = f.temperature.loc[dict(scenario="claimed", config="central", layer=0)]
actual_ch4_t = f.temperature.loc[dict(scenario="actual_ch4", config="central", layer=0)]
claimed_ch4_t = f.temperature.loc[dict(scenario="claimed_ch4", config="central", layer=0)]
actual_cfc12_t = f.temperature.loc[dict(scenario="actual_cfc12", config="central", layer=0)]
claimed_cfc12_t = f.temperature.loc[dict(scenario="claimed_cfc12", config="central", layer=0)]

# --- Full project set temperature ---
plt.figure(figsize=(7, 4))
plt.plot(years, actual - baseline,  label="Actual (tCO2)")
plt.plot(years, claimed - baseline, "--", label="Claimed (total tCO2e)")
plt.axhline(0, linestyle="--", linewidth=1, color='k')
plt.xlim(2000, 2200)
plt.xlabel("Year")
plt.ylabel("ΔT relative to baseline (K)")
plt.title("Total: actual vs claimed temperature impact")
plt.legend()
plt.tight_layout()
plt.show()

# --- CH4 component temperature: actual vs claimed ---
plt.figure(figsize=(7, 4))
plt.plot(years, actual_ch4_t  - baseline, label="Actual (tCH4)")
plt.plot(years, claimed_ch4_t - baseline, "--", label="Claimed (tCO2e)")
plt.axhline(0, linestyle="--", linewidth=1, color='k')
plt.xlim(2000, 2200)
plt.xlabel("Year")
plt.ylabel("ΔT relative to baseline (K)")
plt.title("CH4: actual vs claimed temperature impact")
plt.legend()
plt.tight_layout()
plt.show()

# --- CFC-12 component temperature: actual vs claimed ---
plt.figure(figsize=(7, 4))
plt.plot(years, actual_cfc12_t  - baseline, label="Actual (tCFC-12)")
plt.plot(years, claimed_cfc12_t - baseline, "--", label="Claimed (tCO2e)")
plt.axhline(0, linestyle="--", linewidth=1, color='k')
plt.xlim(2000, 2200)
plt.xlabel("Year")
plt.ylabel("ΔT relative to baseline (K)")
plt.title("CFC-12: actual vs claimed temperature impact")
plt.legend()
plt.tight_layout()
plt.show()

print("done: ran physical and claimed project-flux scenarios")

