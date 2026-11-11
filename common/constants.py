"""
Project-wide constants and default values.

Units used everywhere in the project
    power   : kW
    energy  : kWh
    time    : hours (dt) / integer step index (t)
    price   : Rs/kWh
    SOC     : fraction in [0, 1]
"""

# ----------------------------------------------------------------------
# Time horizon
# ----------------------------------------------------------------------
DT_HOURS = 0.5            # length of one interval
HORIZON_HOURS = 24.0      # simulated horizon
START_HOUR = 6.0          # simulation clock starts at 06:00 and ends 06:00 next day,
                          # so evening arrivals can stay overnight inside the horizon
T_STEPS = int(round(HORIZON_HOURS / DT_HOURS))

# ----------------------------------------------------------------------
# Objective references
# ----------------------------------------------------------------------
H_REF_HOURS = 24.0        # H_ref in J_delay

# ----------------------------------------------------------------------
# EV defaults
# ----------------------------------------------------------------------
DEFAULT_ETA = 0.92
SOC_MIN = 0.0
SOC_MAX = 1.0

BATTERY_CHOICES_KWH = (30.0, 40.0, 60.0, 75.0)
BATTERY_PROBS = (0.30, 0.30, 0.25, 0.15)

P_MAX_CHOICES_KW = (3.3, 7.4, 11.0)          # grid-side on-board charger limits
P_MAX_PROBS = (0.20, 0.50, 0.30)

MIN_STAY_HOURS = 1.0

# ----------------------------------------------------------------------
# Station / grid defaults
# ----------------------------------------------------------------------
STATION_KW_PER_EV = 5.0       # P_m^max = STATION_KW_PER_EV * (#EVs assigned to m)
GRID_CAP_FACTOR = 0.7         # P_grid^max = GRID_CAP_FACTOR * sum_m P_m^max

# Time-of-use tariff by *clock* hour: (start_hour, end_hour, Rs/kWh)
TOU_TARIFF = (
    (0.0, 6.0, 4.5),     # night off-peak
    (6.0, 9.0, 6.0),     # morning shoulder
    (9.0, 17.0, 7.0),    # day
    (17.0, 22.0, 9.5),   # evening peak
    (22.0, 24.0, 5.0),   # late evening
)

# ----------------------------------------------------------------------
# Arrival patterns (clock hours) and demand levels
# ----------------------------------------------------------------------
ARRIVAL_PROFILES = {
    "morning_peak": dict(arrival_mean_h=9.0, arrival_std_h=1.0, stay_mean_h=8.0, stay_std_h=1.5),
    "evening_peak": dict(arrival_mean_h=18.5, arrival_std_h=1.5, stay_mean_h=10.0, stay_std_h=2.0),
    "random":       dict(arrival_mean_h=None, arrival_std_h=None, stay_mean_h=6.0, stay_std_h=2.5),
}

DEMAND_PROFILES = {
    "low":    dict(soc_init=(0.50, 0.70), soc_target=(0.80, 0.85)),
    "normal": dict(soc_init=(0.20, 0.50), soc_target=(0.80, 0.90)),
    "high":   dict(soc_init=(0.05, 0.25), soc_target=(0.90, 0.95)),
}

# Experiment 5 scenarios -> (arrival pattern, demand level)
EXP5_SCENARIOS = {
    "morning_peak": ("morning_peak", "normal"),
    "evening_peak": ("evening_peak", "normal"),
    "random":       ("random", "normal"),
    "high_demand":  ("random", "high"),
    "low_demand":   ("random", "low"),
}

# Experiment 3: (number of EVs, number of stations)
SCALABILITY_CASES = ((10, 1), (25, 2), (50, 3), (100, 5), (200, 10))

TOPOLOGIES = ("ring", "star", "mesh", "random")
ALGORITHMS = (
    "uncoordinated", "centralized",
    "shared_admm", "shared_admm_adaptive",
    "consensus_admm", "consensus_admm_adaptive",
)

NUM_TOL = 1e-9
