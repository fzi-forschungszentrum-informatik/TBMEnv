#!/usr/bin/env python3
"""
"""
from datetime import timedelta

import numpy as np
import pandas as pd
from ray import tune
from scipy.linalg import inv
from tqdm import tqdm

# This is only necessary for one method to tune the
# initial state. However this package has lots and
# lots of dependencies, hence it is optional.
try:
    import hebo
    from ray.tune.search.hebo import HEBOSearch
except ModuleNotFoundError:
    pass


class ThermalModel:
    """
    Thermal model for a concrete core activated office building.

    The implemented RC model follows the approach published in
        P. Zwickel, A. Engelmann, L. Gröll, V. Hagenmeyer, D. Sauer and
        T. Faulwasser "A Comparison of Economic MPC Formulations for Thermal
        Building Control 2019 IEEE PES Innovative Smart Grid Technologies
        Europe (ISGT-Europe) doi: 10.1109/ISGTEurope.2019.8905593.
    henceforth shortly referred to as Zwickel et al.

    The implementation of the thermal model should be mathematically equivalent
    to the Matlab code that has been used in Zwickel et al. and that has been
    provided by the authors for comparison.
    The RC model has been extracted especially from the following functions
    in the Matlab code of Zwickel et al.:
    - generateAMatrix
    - generateSimpleB1Matrix
    - generateB2Matrix

    State Variables (these are computed by the model):
    --------------------------------------------------
    (numbers in brackets refers to the positions in Zwickel et al.'s A matrix)
    (1)  T_roof : Temperature of roof [K].
    (2)  T_cca_s : Temperature in concrete core activation second floor [K].
    (3)  T_zone_s : Zone (air) temperature second floor [K].
    (4)  T_wall_s : Wall temperature second floor [K].
    (5)  T_floor_s : Floor temperature second floor [K].
    (6)  T_cca_f : Temperature in concrete core activation first floor [K].
    (7)  T_zone_f : Zone (air) temperature first floor [K].
    (8)  T_wall_f : Wall temperature first floor [K].
    (9)  T_floor_f : Floor temperature first floor [K].
    (10) T_cca_g : Temperature in concrete core activation first floor [K].
    (11) T_zone_g : Zone (air) temperature ground floor [K].
    (12) T_wall_g : Wall temperature ground floor [K].
    (13) T_floor_g : Floor temperature ground floor [K].
    (14) T_zone_b : Zone (air) temperature ground floor [K].
    (15) T_wall_b : Wall temperature ground floor [K].
    (16) T_floor_b : Floor temperature ground floor [K].

    Controlled Inputs (these must be provided by the optimizer):
    ------------------------------------------------------------
    (numbers in brackets refers to the positions in Zwickel et al.'s B1 matrix)
    (1) phi_cca_s : Heating(+)/cooling(-) power applied to the concrete core
        activated ceiling of the second floor [W].
    (2) phi_rad_s : Heating(+) power applied to the conventional radiators of
        the second floor [W].
    (3) phi_cca_f : Heating(+)/cooling(-) power applied to the concrete core
        activated ceiling of the first floor [W].
    (4) phi_rad_f : Heating(+) power applied to the conventional radiators of
        the first floor [W].
    (5) phi_cca_g : Heating(+)/cooling(-) power applied to the concrete core
        activated ceiling of the ground floor [W].
    (6) phi_rad_g : Heating(+) power applied to the conventional radiators of
        the ground floor [W].

    Disturbances (these come from external data):
    ---------------------------------------------
    (numbers in brackets refers to the positions in Zwickel et al.'s B2 matrix)
    (1) T_out : Outside (ambient) temperature [K].
    (2) T_gnd : Ground temperature [K].
    (3) phi_sol_r : Solar gains of roof [W], i.e.
        combined solar radiation * roof area.
    (4) phi_sol_w : Solar gains of all walls [W], i.e.
        combined solar radiation * wall area.
    (5) phi_ig_s : Internal gains of second floor [W].
    (6) phi_ig_f : Internal gains of first floor [W].
    (7) phi_ig_g : Internal gains of ground floor [W].
    (8) phi_ig_b : Internal gains of basement [W].

    Variables (these must be estimated for the model to work):
    ----------------------------------------------------------
    (numbers in leading brackets indicate position in state tensor,
    names in the brackets refer to the variable names and the values
    after = are the variable values in Zwickel et al.'s Matlab code.)

    (0) A_roof : Area of roof [m^2].
        (Zwickel et al.: obj.data.bldg.A.floor_gfs = 1081.6)
    (1) A_wall_gfs : Area of wall elements of facade on ground/first/second
        floor [m^2].
        (Zwickel et al.: obj.data.bldg.A.wall_gfs = 173.56)
    (2) A_win_gfs : Area of window elements of facade on ground/first/second
        floor [m^2].
        (Zwickel et al.: obj.data.bldg.A.win_gfs = 404.96)

    (3) C_roof : Thermal capacity of roof [J/K].
        (Zwickel et al.: obj.data.bldg.C.roof = 3.7121E+8)
    (4) C_cca : Thermal capacity of concrete core elements [J/K].
        (Zwickel et al.: obj.data.bldg.C.cca_gfs = 6.9333E+8)
    (5) C_zone_gfs : Thermal capacity of zone air [J/K].
        (Zwickel et al.: obj.data.bldg.C.zone_gfs = 1.62246E+7)
    (6) C_zone_b : Thermal capacity of zone air for basement [J/K].
        (Zwickel et al.: obj.data.bldg.C.zone_b = 5.1146E+6)
    (7) C_wall_gfs : Thermal capacity of ground/first/second floor floor [J/K].
        (Zwickel et al.: obj.data.bldg.C.wall_gfs = 8.6986E+7)
    (8) C_wall_b : Thermal capacity of basment floor [J/K].
        (Zwickel et al.: obj.data.bldg.C.wall_b = 1.1809E+8)
    (9) C_floor_fs : Thermal capacity of floor on second and first floor [J/K].
        (Zwickel et al.: obj.data.bldg.C.floor_fs = 3.1475E+7)
    (10) C_floor_g : Thermal capacity of floor on ground floor [J/K].
        (Zwickel et al.: obj.data.bldg.C.floor_g = 1.15162E+9)
    (11) C_floor_b : Thermal capacity of floor for the basement [J/K].
        (Zwickel et al.: obj.data.bldg.C.floor_b = 3.63038E+8)

    (12) G_roof_up : Thermal conductance of roof in upward direction [W/K].
        (Zwickel et al.: obj.data.bldg.G.roof.out = 888.94)
    (13) G_roof_dn : Thermal conductance of roof in downward direction,
        i.e. towards second floor [W/K]..
        (Zwickel et al.: obj.data.bldg.G.roof.in = 371.60) [W/K]..
    (14) G_cca_up : Thermal conductance of the concrete ceiling (above
        the concrete core activation) in up direction, i.e.
        towards next floor/roof [W/K].
        (Zwickel et al.: obj.data.bldg.G.cca_gfs.out = 7996.2).
    (15) G_cca_dn : Thermal conductance of the concrete ceiling (below
        the concrete core activation) in downward direction, i.e. towards
        current floor [W/K].
        (Zwickel et al.: obj.data.bldg.G.cca_gfs.in = 7996.2).
    (16) G_wall_gfs_in : Thermal conductance of the zone walls in inward
        direction, i.e. towards the room [W/K].
        (Zwickel et al.: obj.data.bldg.G.wall_gfs.in = 810.03)
    (17) G_wall_gfs_out : Thermal conductance of the zone walls in outward
        direction, i.e. away from the room [W/K].
        (Zwickel et al.: obj.data.bldg.G.wall_gfs.out = 42.38)
    (18) G_wall_b_in : Thermal conductance of the basement zone walls in inward
        direction, i.e. towards the room [W/K].
        (Zwickel et al.: obj.data.bldg.G.wall_b.in = 1099.69)
    (19 G_wall_b_out : Thermal conductance of the basement zone walls in outward
        direction, i.e. away from the room [W/K].
        (Zwickel et al.: obj.data.bldg.G.wall_b.out = 57.53)
    (20) G_window_fs : Thermal conductance of the windows in first and
        second floor [W/K]. (Zwickel et al.: obj.data.bldg.G.win_gfs = 566.94)
    (21) G_floor_fs_up : Thermal conductance of the floor upwards (into
        the floor), for the first and second floor [W/K].
        (Zwickel et al.: obj.data.bldg.G.floor_fs.in = 1054.44)
    (22) G_floor_fs_dn : Thermal conductance of the floor downwards
        (towards the CCA elements of the floor below), for the first and second
        floor [W/K]. (Zwickel et al.: obj.data.bldg.G.floor_fs.out = 6216.69)
    (23) G_floor_g_up : Thermal conductance of the floor upwards (into
        the floor), for the ground floor [W/K].
        (Zwickel et al.: obj.data.bldg.G.floor_g.in = 409.47)
    (24) G_floor_g_dn : Thermal conductance of the floor downwards for the
        ground floor (towards the ceiling of the basement) [W/K].
        (Zwickel et al.: obj.data.bldg.G.floor_g.out = 1307.96)
    (25) G_floor_b_up : Thermal conductance of the floor upwards (into
        the floor), for the basement [W/K].
        (Zwickel et al.: obj.data.bldg.G.floor_b.in = 129.08)
    (26) G_floor_b_dn : Thermal conductance of the floor downwards for the
        basement (towards the earth) [W/K].
        (Zwickel et al.: obj.data.bldg.G.floor_b.out = 412.32)

    (27) eta_cca: Efficiency of heat transfer between heating/cooling water and
        concrete core activated ceiling.
        (Zwickel et al.: obj.data.constants.eta_cca = 1.0)
    (28) eta_rad: Efficiency of heat transfer between heating water and
        radiators. (Zwickel et al.: obj.data.constants.eta_rad = 1.0)

    (29) f_sol_roof : Solar gains coefficient of the roof.
        (Zwickel et al.: obj.data.bldg.f.sol_roof = 0.05)
    (30) f_sol_win_gfs : Solar gains coefficient of the windows.
        (Zwickel et al.: obj.data.bldg.f.sol_win_gfs = 0.28665)
    (31) f_sol_wall_gfs : Solar gains coefficient of the walls.
        (Zwickel et al.: obj.data.bldg.f.sol_wall_gfs = 0.015)
    (32) f_vent_gfs : Thermal heat transfer factor for air exchange
        (ventilation) for the ground/first/second floor.
        (Zwickel et al.: obj.data.bldg.f.vent_gfs = 630.957)
    (33) f_vent_b : Thermal heat transfer factor for air exchange (ventilation)
        for the basement.
        (Zwickel et al.: obj.data.bldg.f.vent_b = 198.903)
    (34) f_ground: ???
        (Zwickel et al.: obj.data.bldg.f.ground = 0.68475)
    (35) f_b : Thermal heat transfer factor for the basement (why? no clue).
        (Zwickel et al.: obj.data.bldg.f.basement = 0.31524)
    """

    # This is just a little utility that might come in handy to generate
    # the `variables` and `initial_state` tensors from a dict or something
    # else for which order not guaranteed.
    variable_names_ordered = [
        "A_roof",
        "A_wall_gfs",
        "A_win_gfs",
        "C_roof",
        "C_cca",
        "C_zone_gfs",
        "C_zone_b",
        "C_wall_gfs",
        "C_wall_b",
        "C_floor_fs",
        "C_floor_g",
        "C_floor_b",
        "G_roof_up",
        "G_roof_dn",
        "G_cca_up",
        "G_cca_dn",
        "G_wall_gfs_in",
        "G_wall_gfs_out",
        "G_wall_b_in",
        "G_wall_b_out",
        "G_window_fs",
        "G_floor_fs_up",
        "G_floor_fs_dn",
        "G_floor_g_up",
        "G_floor_g_dn",
        "G_floor_b_up",
        "G_floor_b_dn",
        "eta_cca",
        "eta_rad",
        "f_sol_roof",
        "f_sol_win_gfs",
        "f_sol_wall_gfs",
        "f_vent_gfs",
        "f_vent_b",
        "f_ground",
        "f_b",
    ]
    initial_state_names_ordered = [
        "T_roof",
        "T_cca_s",
        "T_zone_s",
        "T_wall_s",
        "T_floor_s",
        "T_cca_f",
        "T_zone_f",
        "T_wall_f",
        "T_floor_f",
        "T_cca_g",
        "T_zone_g",
        "T_wall_g",
        "T_floor_g",
        "T_zone_b",
        "T_wall_b",
        "T_floor_b",
    ]
    disturbance_names_ordered = [
        "T_out",
        "T_gnd",
        "phi_sol_r",
        "phi_sol_w",
        "phi_ig_s",
        "phi_ig_f",
        "phi_ig_g",
        "phi_ig_b",
    ]

    def __init__(
        self,
        weather_data,
        d_t=60,
        n_fd_steps=15,
        variables=None,
        initial_state=None,
        phi_fraction=None,
        phi_ig_base=None,
    ):
        """
        Init thermal model.

        Arguments:
        ----------
        weather_data : pandas.DataFrame
            A dataframe containing the ambient outside temperature [°C] in
            column `Temperature` and global solar irradicance [W/m^2]
            in column `Irradiance`. Note that the possible time range the
            thermal model can simulate will be limited to time range between
            the first and the last entry of the index of this dataframe.
            Further note that this dataframe should be in 15 minute resolution
            or better for best results (but definitly *not* daily values
            or worse).
        d_t : float
            The finite difference step size used for simulation in seconds.
            Defaults to 60s.
        n_fd_steps : integer
            The number of finite difference steps per model step. Defaults
            to 15. The simulated time between subsequent calls to `self.step`
            is `d_t * n_fd_steps` in seconds.
        variables : numpy.ndarray
            Variables vector of shape `(36,)`. See details in class docstring.
            If None, defaults to the variable values used by Zwickel et al.
        initial_state : numpy.ndarray
            State vector of shape `(16,)`. See details in class docstring.
            If None, defaults to the state values used by Zwickel et al.
        phi_fraction : dict of float values.
            Measurements of the real building do only contain heating/cooling
            power values for all three floors combined. During optimizing the
            initial state however, we need those explicitly for each floor.
            This is the mapping of how of the total cooling/heating power is
            distributed to each floor. Must contain one key for each entry
            in "Controlled Inputs " in class docstring. Defaults to None in
            which case the cooling/heating power is distributed evenly. See
            code for details.
        phi_ig_base : dict of float values.
            Baseload of the internal gains. I.e. it will be assumed that this
            power is applied to the zones at every step. Keys must match the
            items (5) to (8) of the "Disturbances" listed above. Defaults to
            None in the values are assumed to be zero. See code for details.
        """
        self.d_t = d_t
        self.n_fd_steps = n_fd_steps

        # Preprocess weather data. This model computes internally with all SI
        # units, hence adapt the temperatures to Kelvin.
        weather_data = weather_data.copy()
        weather_data["Temperature"] += 273.15
        # Furthermore we resample the weather_data dataframe such that the
        # index will contain an entry for every timestep that may be simulated
        # based on weather_data. This approach has the caveat that it keeps
        # all the interpolated values in memory (which might be a lot for long
        # horizons). On the other hand simulation should be faster as
        # interpolation on the fly might be tedious.
        weather_data = weather_data.resample(timedelta(seconds=d_t)).mean()
        # Now all intermediate values should be NaN. Hence fill them.
        weather_data = weather_data.interpolate(method="linear")
        # Expose to self.compute_disturbance
        self.weather_data = weather_data

        if variables is not None:
            self.variables = variables
        else:
            self.variables = np.asarray(
                [
                    1081.6,  # A_roof
                    173.56,  # A_wall_gfs
                    404.96,  # A_win_gfs
                    3.7121e8,  # C_roof
                    6.9333e8,  # C_cca
                    1.62246e7,  # C_zone_gfs
                    5.1146e6,  # C_zone_b
                    8.6986e7,  # C_wall_gfs
                    1.1809e8,  # C_wall_b
                    3.1475e7,  # C_floor_fs
                    1.15162e9,  # C_floor_g
                    3.63038e8,  # C_floor_b
                    888.94,  # G_roof_up
                    371.60,  # G_roof_dn
                    7996.2,  # G_cca_up
                    7996.2,  # G_cca_dn
                    810.03,  # G_wall_gfs_in
                    42.38,  # G_wall_gfs_out
                    1099.69,  # G_wall_b_in
                    57.53,  # G_wall_b_out
                    566.94,  # G_window_fs
                    1054.44,  # G_floor_fs_up
                    6216.69,  # G_floor_fs_dn
                    409.47,  # G_floor_g_up
                    1307.96,  # G_floor_g_dn
                    129.08,  # G_floor_b_up
                    412.32,  # G_floor_b_dn
                    1.0,  # eta_cca
                    1.0,  # eta_rad
                    0.05,  # f_sol_roof
                    0.28665,  # f_sol_win_gfs
                    0.015,  # f_sol_wall_gfs
                    630.957,  # f_vent_gfs
                    198.903,  # f_vent_b
                    0.68475,  # f_ground
                    0.31524,  # f_b
                ]
            )

        if initial_state is not None:
            self.initial_state = initial_state
        else:
            self.initial_state = np.asarray(
                [
                    299.34,  # T_roof
                    294.15,  # T_cca_s
                    294.15,  # T_zone_s
                    294.38,  # T_wall_s
                    294.15,  # T_floor_s
                    294.15,  # T_cca_f
                    294.15,  # T_zone_f
                    294.38,  # T_wall_f
                    294.15,  # T_floor_f
                    294.15,  # T_cca_g
                    294.15,  # T_zone_g
                    294.38,  # T_wall_g
                    286.51,  # T_floor_g
                    285.79,  # T_zone_b
                    285.79,  # T_wall_b
                    285.78,  # T_floor_b
                ]
            )

        if phi_fraction is not None:
            self.phi_fraction = phi_fraction
        else:
            self.phi_fraction = {
                "phi_cca_s": 0.333,
                "phi_rad_s": 0.333,
                "phi_cca_f": 0.333,
                "phi_rad_f": 0.333,
                "phi_cca_g": 0.333,
                "phi_rad_g": 0.333,
            }

        if phi_ig_base is not None:
            self.phi_ig_base = phi_ig_base
        else:
            self.phi_ig_base = {
                "phi_ig_s": 0.0,
                "phi_ig_f": 0.0,
                "phi_ig_g": 0.0,
                "phi_ig_b": 0.0,
            }

        matrices = self.create_state_space_matrices_from_variables(
            self.variables
        )
        self.A, self.B1, self.B2 = matrices

    def create_state_space_matrices_from_variables(self, variables):
        """
        Computes the A, B1 and B2 matrices from variables.

        Arguments:
        ----------
        variables : numpy.ndarray
            Variables vector of shape `(36,)`. See details in class docstring.

        Returns:
        --------
        A : np.array
            State transition matrix with shape (16, 16)
        B1 : np.array
            Input matrix with shape (16, 6)
        B2 : np.array
            Disturbance matrix with shape (16, 8)
        """
        A = np.zeros((17, 17))
        B1 = np.zeros((17, 7))
        B2 = np.zeros((17, 9))

        C_roof = variables[3]
        C_cca = variables[4]
        C_zone_gfs = variables[5]
        C_zone_b = variables[6]
        C_wall_gfs = variables[7]
        C_wall_b = variables[8]
        C_floor_fs = variables[9]
        C_floor_g = variables[10]
        C_floor_b = variables[11]
        G_roof_up = variables[12]
        G_roof_dn = variables[13]
        G_cca_up = variables[14]
        G_cca_dn = variables[15]
        G_wall_gfs_in = variables[16]
        G_wall_gfs_out = variables[17]
        G_wall_b_in = variables[18]
        G_wall_b_out = variables[19]
        G_window_fs = variables[20]
        G_floor_fs_up = variables[21]
        G_floor_fs_dn = variables[22]
        G_floor_g_up = variables[23]
        G_floor_g_dn = variables[24]
        G_floor_b_up = variables[25]
        G_floor_b_dn = variables[26]
        eta_cca = variables[27]
        eta_rad = variables[28]
        f_sol_roof = variables[29]
        f_sol_win_gfs = variables[30]
        f_sol_wall_gfs = variables[31]
        f_vent_gfs = variables[32]
        f_vent_b = variables[33]
        f_ground = variables[34]
        f_b = variables[35]

        # NOTE: This is not zero indexed as in Python.

        # Roof.
        A[1, 1] = -(G_roof_up + G_roof_dn + G_cca_up) / C_roof
        A[1, 2] = (G_roof_dn + G_cca_up) / C_roof
        B2[1, 1] = G_roof_up / C_roof
        B2[1, 3] = f_sol_roof / C_roof

        # CCA second floor.
        A[2, 1] = (G_roof_dn + G_cca_up) / C_cca
        A[2, 2] = -(G_roof_dn + G_cca_up + G_cca_dn) / C_cca
        A[2, 3] = G_cca_dn / C_cca
        B1[2, 1] = eta_cca / C_cca

        # Zone second floor.
        A[3, 2] = G_cca_dn / C_zone_gfs
        A[3, 3] = (
            -(
                G_cca_dn
                + G_wall_gfs_in
                + G_floor_fs_up
                + G_window_fs
                + f_vent_gfs
            )
            / C_zone_gfs
        )
        A[3, 4] = G_wall_gfs_in / C_zone_gfs
        A[3, 5] = G_floor_fs_up / C_zone_gfs
        B1[3, 2] = eta_rad / C_zone_gfs

        # Note: This equation (and the ones for the other floors too) is
        # probably wrong as f_vent_gfs should rather be G_vent_gfs, as
        # ventilation is marked as a thermal resistance in Zwickel et al.
        # Furthermore G_windows_* and G_vent_* are marked to be parallel
        # resistances. Hence the equation must likely look like this:
        # ... `+= T_out * 1/(1/G_window_fs + 1/G_vent_gfs) / C_zone_gfs`
        B2[3, 1] = (G_window_fs + f_vent_gfs) / C_zone_gfs
        B2[3, 4] = f_sol_win_gfs / C_zone_gfs
        B2[3, 5] = 1 / C_zone_gfs

        # Walls second floor.
        A[4, 3] = G_wall_gfs_in / C_wall_gfs
        A[4, 4] = -(G_wall_gfs_in + G_wall_gfs_out) / C_wall_gfs
        B2[4, 1] = G_wall_gfs_out / C_wall_gfs
        B2[4, 4] = f_sol_wall_gfs / C_wall_gfs

        # Floor second floor.
        A[5, 3] = G_floor_fs_up / C_floor_fs
        A[5, 5] = -(G_floor_fs_up + G_floor_fs_dn + G_cca_up) / C_floor_fs
        A[5, 6] = (G_floor_fs_dn + G_cca_up) / C_floor_fs

        # CCA first floor.
        A[6, 5] = (G_floor_fs_dn + G_cca_up) / C_cca
        A[6, 6] = -(G_floor_fs_dn + G_cca_up + G_cca_dn) / C_cca
        A[6, 7] = G_cca_dn / C_cca
        B1[6, 3] = eta_cca / C_cca

        # Zone first floor.
        A[7, 6] = G_cca_dn / C_zone_gfs
        A[7, 7] = (
            -(
                G_cca_dn
                + G_wall_gfs_in
                + G_floor_fs_up
                + G_window_fs
                + f_vent_gfs
            )
            / C_zone_gfs
        )
        A[7, 8] = G_wall_gfs_in / C_zone_gfs
        A[7, 9] = G_floor_fs_up / C_zone_gfs
        B1[7, 4] = eta_rad / C_zone_gfs
        B2[7, 1] = (G_window_fs + f_vent_gfs) / C_zone_gfs
        B2[7, 4] = f_sol_win_gfs / C_zone_gfs
        B2[7, 6] = 1 / C_zone_gfs

        # Walls first floor.
        A[8, 7] = G_wall_gfs_in / C_wall_gfs
        A[8, 8] = -(G_wall_gfs_in + G_wall_gfs_out) / C_wall_gfs
        B2[8, 1] = G_wall_gfs_out / C_wall_gfs
        B2[8, 4] = f_sol_wall_gfs / C_wall_gfs

        # Floor first floor.
        A[9, 7] = G_floor_fs_up / C_floor_fs
        A[9, 9] = -(G_floor_fs_up + G_floor_fs_dn + G_cca_up) / C_floor_fs
        A[9, 10] = (G_floor_fs_dn + G_cca_up) / C_floor_fs

        # CCA ground floor.
        A[10, 9] = (G_floor_fs_dn + G_cca_up) / C_cca
        A[10, 10] = -(G_floor_fs_dn + G_cca_up + G_cca_dn) / C_cca
        A[10, 11] = G_cca_dn / C_cca
        B1[10, 5] = eta_cca / C_cca

        # Zone ground floor.
        A[11, 10] = G_cca_dn / C_zone_gfs
        A[11, 11] = (
            -(
                G_cca_dn
                + G_wall_gfs_in
                + G_floor_g_up
                + G_window_fs
                + f_vent_gfs
            )
            / C_zone_gfs
        )
        A[11, 12] = G_wall_gfs_in / C_zone_gfs
        A[11, 13] = G_floor_g_up / C_zone_gfs
        B1[11, 6] = eta_rad / C_zone_gfs
        B2[11, 1] = (G_window_fs + f_vent_gfs) / C_zone_gfs
        B2[11, 4] = f_sol_win_gfs / C_zone_gfs
        B2[11, 7] = 1 / C_zone_gfs

        # Walls ground floor.
        A[12, 11] = G_wall_gfs_in / C_wall_gfs
        A[12, 12] = -(G_wall_gfs_in + G_wall_gfs_out) / C_wall_gfs
        B2[12, 1] = G_wall_gfs_out / C_wall_gfs
        B2[12, 4] = f_sol_wall_gfs / C_wall_gfs

        # Floor ground floor.
        A[13, 11] = G_floor_g_up / C_floor_g
        A[13, 13] = -(G_floor_g_up + G_floor_g_dn) / C_floor_g
        A[13, 14] = (f_b * G_floor_g_dn) / C_floor_g
        B2[13, 2] = (f_ground * G_floor_g_dn) / C_floor_g

        # Zone basement.
        A[14, 13] = (f_b * G_floor_g_dn) / C_zone_b
        A[14, 14] = (
            -(f_b * G_floor_g_dn + G_wall_b_in + G_floor_b_up + f_vent_b)
            / C_zone_b
        )
        A[14, 15] = G_wall_b_in / C_zone_b
        A[14, 16] = G_floor_b_up / C_zone_b
        B2[14, 1] = f_vent_b / C_zone_b
        B2[14, 8] = 1 / C_zone_b

        # Walls basement.
        A[15, 14] = G_wall_b_in / C_wall_b
        A[15, 15] = -(G_wall_b_in + G_wall_b_out) / C_wall_b
        B2[15, 2] = G_wall_b_out / C_wall_b

        # Floor basement.
        A[16, 14] = G_floor_b_up / C_floor_b
        A[16, 16] = -(G_floor_b_up + G_floor_b_dn) / C_floor_b
        B2[16, 2] = G_floor_b_dn / C_floor_b

        # Remove the first entry to get something zero indexed.
        A = A[1:, 1:]
        B1 = B1[1:, 1:]
        B2 = B2[1:, 1:]

        return A, B1, B2

    def tune_initial_state(
        self,
        bd,
        n_suggestions=10,
        n_observed=10,
        n_sim_steps=96,
        show_progressbar=False,
    ):
        """
        Tunes the intial state such that it matches observations.

        This uses Bayesian Optimization to optimize `self.initial_state`

        Arguments:
        ----------
        bd : pandas.DataFrame
            The building data used for optimization. Index must contain
            datetime entries matching the model steps. Must contain the columns
            `phi_rad` holding the heating power sum exchanged with the radiators
             and `phi_cca` holding the heating/cooling power sum exchanged with
            the concrete core elements.
        n_suggestions : int
            Number of parallel candidates sampled from HEBO. The default of 10
            should be reasonable trade-off between total number of observed
            candidates and runtime.
        n_observed : int
            How many observations are presented to HEBO. The total number of
            evaluations (forward passes through the model) is
            `n_observed * n_suggestions`. More is generally better, but will
            scale sub linear to execution time.
        n_sim_steps : int
            Home many model steps should be be simulated forward before
            computing the loss value.
        """

        # Search a space between 0°C and 45°C for the best intial states.
        search_space = hebo.design_space.design_space.DesignSpace()
        raw_search_space = []
        for initial_state_name in self.initial_state_names_ordered:
            raw_search_space.append(
                {
                    "name": initial_state_name,
                    "type": "num",
                    "lb": 273.15,
                    "ub": 318.15,
                },
            )
        search_space.parse(raw_search_space)

        optimizer = hebo.optimizers.hebo.HEBO(search_space)
        # Store the optimizer, one might want to inspect it for debugging.
        self._initial_state_optimizer = optimizer

        if show_progressbar:
            pbar_total = n_observed * n_suggestions
            progressbar = tqdm(total=pbar_total, desc="Tuning intial state")

        for _ in range(n_observed):
            # This is a Dataframe with the temperatures as columns.
            # Use multiple suggestions at once as the slow part is acutally
            # fitting the GP after the data has been observed, while the
            # forwarded passes through the model are much faster.
            losses = []
            suggested_initial_states_df = optimizer.suggest(
                n_suggestions=n_suggestions
            )
            for (
                _,
                suggested_initial_state_row,
            ) in suggested_initial_states_df.iterrows():

                # Store as initial_state for the next simulation run.
                self.initial_state = suggested_initial_state_row.values

                # Run simulation forward for requested number of steps.
                # This will likely fail if the indexes of bd and
                # `self.weather_data` are not well aligned.
                _ = self.reset()
                states_and_dones = self.step_many(
                    actions=bd[["phi_cca", "phi_rad"]].iloc[:n_sim_steps],
                )
                if len(states_and_dones) != n_sim_steps:
                    print(
                        "WARNING: tune_initial_state was requested to perform ",
                        "{} forward step but did {} instead.".format(
                            n_sim_steps, len(states_and_dones)
                        ),
                    )

                # Compute loss (states_and_dones are expected in K, bd in °C)
                residuals = states_and_dones - bd - 273.15
                # Remove all columns apart from the zone temperatures
                # (they are filled with NaNs).
                residuals = residuals.dropna(how="all", axis=1).dropna(
                    how="all"
                )
                rmse_loss = np.square(residuals.values).mean()
                losses.append(rmse_loss)

                if show_progressbar:
                    progressbar.update(1)

            optimizer.observe(
                suggested_initial_states_df,
                np.asarray(losses),
            )

            if show_progressbar:
                progressbar.set_postfix(RMSE=optimizer.y.min())

        if show_progressbar:
            progressbar.close()

        # Finally set the initial state to the best found configuration.
        self.initial_state = optimizer.X.iloc[optimizer.y.argmin()].values
        self.reset()

    def compute_current_disturbance(self, current_dt):
        """
        Compute the values of the disturbances for a given datetime object.

        Arguments:
        ----------
        current_dt : datetime.datetime object
            The date and time for which the disturbances should be computed.

        Returns:
        --------
        current_disturbance : dict of float.
            Disturbance values. See details in class docstring.

        Raises:
        -------
        KeyError:
            if `current_dt` is not in the index of `self.weather_data`.
        """
        current_temperature = self.weather_data["Temperature"][current_dt]
        current_irradiance = self.weather_data["Irradiance"][current_dt]

        current_disturbance = {
            "T_out": current_temperature,
            # Technically the ground temperature is not equivalent to the air
            # temperature, however as we have no on-site measurements of ground
            # temperature we use air temperature instead, which should be
            # roughly similar.
            "T_gnd": current_temperature,
            # Irradiance * Roof area
            "phi_sol_r": current_irradiance * self.variables[0],
            # This is not very nice here, we throw the window area and wall
            # area together, just to let divide it back later with the
            # f_sol_win_gfs and f_sol_wall_gfs coefficients. But well that is
            # how Zwickel et al. have implemented the model.
            "phi_sol_w": (
                current_irradiance * (self.variables[1] + self.variables[2])
            ),
            # Fixed internal gains, should use a weekly cycle instead.
            "phi_ig_s": self.phi_ig_base["phi_ig_s"],
            "phi_ig_f": self.phi_ig_base["phi_ig_f"],
            "phi_ig_g": self.phi_ig_base["phi_ig_g"],
            "phi_ig_b": self.phi_ig_base["phi_ig_b"],
        }

        return current_disturbance

    def simulate_forward(self, state, action, initial_dt):
        """
        Run the RC simulation model forward.

        Uses finite differences approximation for integration.

        Arguments:
        ----------
        state : numpy.ndarray
            State vector, must be of shape `(16,)`.
            See details in class docstring.
        action : numpy.ndarray
            Action vector with `shape == (6,)`.
            See details in class docstring.
        initial_dt : datetime.datetime object
            The time and date of the when the first simulation step starts.
            Is used to compute the disturbances per step.

        Returns:
        --------
        state_after_model_step : numpy.ndarray
            Similar to state but containing the state values at the end of
            the simulated time, i.e. after taking the last finite difference
            step.
        dt_after_model_step : float
            The time the simulation has passed forward in seconds. Assuming
            the default values this would be `60 * 15 = 900` seconds.
        done : bool
            Is true if this is the last step that can be simulated, i.e.
            the last entry in `self.weather_data` has been reached.
        """
        done = False

        current_dt = initial_dt
        state_after_model_step = state
        for _ in range(self.n_fd_steps):
            # Compute the disturbances for the current simulated time step.
            disturbance_dict = self.compute_current_disturbance(current_dt)
            disturbance = np.asarray(
                [disturbance_dict[k] for k in self.disturbance_names_ordered]
            )

            # Update the state vector
            state_after_model_step = (
                state_after_model_step
                + (
                    self.A @ state_after_model_step
                    + self.B1 @ action
                    + self.B2 @ disturbance
                )
                * self.d_t
            )

            # Update the datetime for the disturbances of the next step.
            current_dt += timedelta(seconds=self.d_t)

            # check if this has been the last entry in `self.weather_data`
            if current_dt >= self.weather_data.index[-1]:
                done = True

            # Break after the update, just in case somebody might further use
            # dt_after_model_step, then this ensures that it is really the
            # datetime after the model step.
            if done:
                break

        #################################################################
        # Prepare the output.
        #################################################################
        dt_after_model_step = current_dt

        return state_after_model_step, dt_after_model_step, done

    def reset(self):
        """
        Reset the model and simulation to the initial state.

        Returns:
        --------
        initial_state : numpy.ndarray
            Same as defined in self.__init__
        """
        self.state = self.initial_state
        self.current_dt = self.weather_data.index[0]

        return self.state

    def step(self, action):
        """
        Apply action to the the thermal model for one model step.

        Arguments:
        ----------
        action : numpy.ndarray
            Action vector of `shape == (6,)`.
            See details in class docstring.

        Returns:
        --------
        state : numpy.ndarray
            State vector of shape `(16,)`.
            See details in class docstring.
        done : bool
            Is true if this is the last step that can be simulated, i.e.
            the last entry in `self.weather_data` has been reached.
        """

        sr = self.simulate_forward(self.state, action, self.current_dt)
        state_after_model_step, dt_after_model_step, done = sr

        self.state = state_after_model_step
        self.current_dt = dt_after_model_step

        return self.state, done

    def step_many(self, actions, show_progressbar=False):
        """
        Similar to step, but allows to execute a sequence of actions.

        This method is especially intended for simulating larger time spans
        to compare against measured data. Hence, all inputs/outputs are
        dataframes to simplify handling and to ensure that the action matches
        the weather data etc.

        Note: You must have called reset at least once before calling this
        method.

        Arguments:
        ----------
        actions : pandas.DataFrame
            Must contain `phi_cca` and `phi_rad` entries. Distribution of
            these values to the floors is made using `self.phi_fraction`.
        show_progressbar : bool
            If true show a progressbar assuming that total number of steps
            will be the length of `actions`. Defaults to False.

        Returns:
        --------
        states_and_dones : pandas.DataFrame
            All states and corresponding done values in a single dataframe.
        """
        current_dts = [self.current_dt]
        states = [self.state]
        dones = [False]

        # While loop iterates until we have it is not possible to retrieve
        # an action for the current timestep of the simulation any more.
        if show_progressbar:
            progressbar = tqdm(total=len(actions) - 1, desc="Simulating steps")
        while self.current_dt in actions.index:
            action_cdt = actions.loc[self.current_dt]
            action = np.asarray(
                [
                    action_cdt["phi_cca"] * self.phi_fraction["phi_cca_s"],
                    action_cdt["phi_rad"] * self.phi_fraction["phi_rad_s"],
                    action_cdt["phi_cca"] * self.phi_fraction["phi_cca_f"],
                    action_cdt["phi_rad"] * self.phi_fraction["phi_rad_f"],
                    action_cdt["phi_cca"] * self.phi_fraction["phi_cca_g"],
                    action_cdt["phi_rad"] * self.phi_fraction["phi_rad_g"],
                ]
            )

            # Note that self.current_dt changes while calling step.
            state, done = self.step(action)
            current_dts.append(self.current_dt)
            states.append(state)
            dones.append(done)

            if show_progressbar:
                progressbar.update(1)

            # This breaks the loop once weather data has reached an end.
            if done:
                break

        if show_progressbar:
            progressbar.close()

        states = np.asarray(states)
        states_and_dones = pd.DataFrame(
            index=current_dts,
            data=states,
            columns=self.initial_state_names_ordered,
        )
        states_and_dones["done"] = np.asarray(dones)

        return states_and_dones


class Baseline1:
    """
    Baseline for Scenario 1 using the thermal model above.
    """

    def __init__(self):
        self.obs_all = pd.DataFrame()
        self.actions_all = pd.DataFrame()
        self.disturbances_all = pd.DataFrame()
        self.n_predictions = 0

        # Use default parameters of `ThermalModel` if optimized
        # values are not found yet.
        self.variables = None
        self.phi_fraction = None
        self.phi_ig_base = None

    def observe(self, obs, actions, disturbances):
        """
        Just stores the data for training.

        Arguments:
        ----------
        obs : pandas.DataFrame or list of pandas.DataFrame
            The observation values that should be added to the
            training data of the model.
        actions : pandas.DataFrame or list of pandas.DataFrame
            Like `obs` but the setpoints.
        disturbances : pandas.DataFrame or list of pandas.DataFrame
            Like `obs` but for the predictions of disturbances.
        """
        if isinstance(obs, list):
            obs = pd.concat(obs, axis=0)
        if isinstance(actions, list):
            actions = pd.concat([a.iloc[[0]] for a in actions], axis=0)
        else:
            self.last_action_sequence = actions
            actions = actions.iloc[[0]]
        if isinstance(disturbances, list):
            disturbances = pd.concat(
                [d.iloc[[0]] for d in disturbances], axis=0
            )
        else:
            self.last_disturbance_forecast = disturbances
            disturbances = disturbances.iloc[[0]]

        self.obs_all = pd.concat([self.obs_all, obs])
        self.actions_all = pd.concat([self.actions_all, actions])
        self.disturbances_all = pd.concat([self.disturbances_all, disturbances])

    def estimate_initial_state_extrapolate_obs(self, obs_dt=None):
        """
        A very crude method for estimating the initial state.

        This just assume that walls etc. have the same temperature
        as the corresponding zone temperature.

        Arguments:
        ----------
        obs_dt : int
            The timestamp (of `self.obs_all`) for which the state should
            be estimated. If `None` defaults to the last one. This is
            usually what you want.

        Returns:
        --------
        initial_state : numpy.ndarray
            Estimate of state vector with shape `(16,)`.
        """
        if obs_dt is None:
            obs_dt = self.obs_all.index[-1]
        T_zone_s = self.obs_all.loc[obs_dt]["T_zone_s"]
        T_zone_f = self.obs_all.loc[obs_dt]["T_zone_f"]
        T_zone_g = self.obs_all.loc[obs_dt]["T_zone_g"]

        # Assumes basement has similar temperature then ground.
        initial_state = np.asarray(
            [T_zone_s] * 5 + [T_zone_f] * 4 + [T_zone_g] * 7
        )

        # Convert from °C to K.
        initial_state += 273.15

        return initial_state

    def estimate_initial_state_kalman_filter(
        self, obs_dt=None, n_kalman_steps=6
    ):
        """
        Estimate the initial state using a Kalman Filter.

        Arguments:
        ----------
        obs_dt : int
            The timestamp (of `self.obs_all`) for which the state should
            be estimated. If `None` defaults to the last one. This is
            usually what you want.
        n_kalman_steps : int
            The number of previous iterations the Kalman Filter should
            do, i.e. how many previous observations are used.

        Returns:
        --------
        initial_state : numpy.ndarray
            Estimate of state vector with shape `(16,)`.
        """
        if obs_dt is None:
            obs_dt = self.obs_all.index[-1]

        # NOTE: The last ts is the one we want to approach, hence we need to
        # have one more ts then steps.
        i_from = -1 - n_kalman_steps
        ts_range = self.obs_all.loc[:obs_dt].iloc[i_from:].index

        # The inital state estimate to start with.
        x = self.estimate_initial_state_extrapolate_obs(obs_dt=ts_range[0])

        # Assume that we are rather uncertain, i.e std of 2.0 about the
        # estimated states, but very certain, i.e. std of 0.1 about the
        # measured values
        variances = np.full(16, 4.0)
        variances[2] = variances[6] = variances[10] = 0.01
        P = np.zeros((16, 16))
        np.fill_diagonal(P, variances)

        # Definition of Process noise Q for the Kalman Filter
        # NOTE: This is a very simplyfied approach to process noise
        # A more sufficticated approach is provided here:
        # https://link.springer.com/content/pdf/10.1007/978-3-658-16728-8_7.pdf
        Q = np.zeros((16, 16))
        np.fill_diagonal(Q, 0.1)

        # observation noise
        R = np.zeros((3, 3))
        np.fill_diagonal(R, 0.1)

        H = np.zeros((3, 16))
        # Set the values of A for the desired indices
        H[0, 2] = 1
        H[1, 6] = 1
        H[2, 10] = 1

        # The thermal model used for estimating next steps.
        tm = ThermalModel(
            weather_data=self.disturbances_all.loc[ts_range],
            initial_state=x,
            variables=self.variables,
            phi_fraction=self.phi_fraction,
            phi_ig_base=self.phi_ig_base,
        )
        actions = self.actions_all.loc[ts_range]
        actions_tm = pd.concat(
            [
                actions["P_cool_cca"] * tm.phi_fraction["phi_cca_s"],
                actions["P_cool_cca"] * 0,  # No heating.
                actions["P_cool_cca"] * tm.phi_fraction["phi_cca_f"],
                actions["P_cool_cca"] * 0,
                actions["P_cool_cca"] * tm.phi_fraction["phi_cca_g"],
                actions["P_cool_cca"] * 0,
            ],
            axis=1,
        )

        for i in range(n_kalman_steps):
            ts = ts_range[i]
            x_dash_plus_1, _, _ = tm.simulate_forward(
                state=x,
                action=actions_tm.loc[ts].values,
                initial_dt=ts,
            )
            # Select the measured part of the state variables.
            z_plus_1 = H @ x_dash_plus_1
            # Compute innovation, note z is in K, obs in °C
            v = self.obs_all.loc[ts_range[i + 1]] - (z_plus_1 - 273.15)

            # Covariance of the predicted state `x_dash_plus_1`
            P_dash_plus_1 = tm.A @ P @ tm.A.transpose() + Q

            # Covariance of the innovation.
            S = H @ P_dash_plus_1 @ H.transpose() + R

            # Compute the Kalman Gain
            W = P_dash_plus_1 @ H.transpose() @ inv(S, overwrite_a=True)

            # Correction of the covariance.
            P_plus_1 = (np.eye(16, 16) - W @ H) @ P_dash_plus_1

            # Correction of the state vector
            x_plus_1 = x_dash_plus_1 + W @ v

            # Prepare for the next iteration
            P = P_plus_1
            x = x_plus_1

        initial_state = x

        return initial_state

    def predict(self, use_kalman_filter=True, tune_model=True):
        """
        Triggers a prediction for the timestamp for which data is available.

        Returns:
        --------
        state_prediction: pandas.DataFrame
            The prediction of future observed state variables.
        """
        if tune_model and self.n_predictions % 672 == 0:
            best_config = self.tune_variables()
            best_config = self.tm_args_from_config(best_config)
            self.variables = best_config[0]
            self.phi_fraction = best_config[1]
            self.phi_ig_base = best_config[2]
        self.n_predictions += 1

        if use_kalman_filter:
            initial_state = self.estimate_initial_state_kalman_filter()
        else:
            initial_state = self.estimate_initial_state_extrapolate_obs()

        tm = ThermalModel(
            weather_data=self.last_disturbance_forecast,
            initial_state=initial_state,
            variables=self.variables,
            phi_fraction=self.phi_fraction,
            phi_ig_base=self.phi_ig_base,
        )
        tm.reset()
        actions_tm = pd.DataFrame(
            index=self.last_action_sequence.index,
            data={
                "phi_cca": self.last_action_sequence["P_cool_cca"] * 1000,
                "phi_rad": np.zeros(len(self.last_action_sequence)) * 1000,
            },
        )
        states_and_dones = tm.step_many(actions=actions_tm)

        # Extract the observed states and convert back to °C
        state_prediction = (
            states_and_dones[["T_zone_s", "T_zone_f", "T_zone_g"]] - 273.15
        )
        # Remove the first state, as it is the same time as the one
        # we have observed the state already.
        state_prediction = state_prediction.iloc[1:]
        return state_prediction

    @staticmethod
    def tm_args_from_config(config):
        """
        Transforms the output of tune algorithm to the input for `ThermalModel`.

        Arguments:
        ----------
        config : dict
            As returned by tune.

        Returns:
        --------
        variables : numpy.ndarray
            See `ThermalModel`.
        phi_fraction : dict of float values.
            See `ThermalModel`.
        phi_ig_base : dict of float values.
            See `ThermalModel`.
        """
        variables = np.asarray(
            [config[v] for v in ThermalModel.variable_names_ordered]
        )

        # Don't optimize the values for the radiators. There should be
        # no heating days in the data.
        phi_fraction = {
            "phi_cca_s": config["frac_phi_cca_s"],
            "phi_rad_s": 0.333,
            "phi_cca_f": config["frac_phi_cca_f"],
            "phi_rad_f": 0.333,
            "phi_cca_g": 1.0
            - config["frac_phi_cca_s"]
            - config["frac_phi_cca_f"],
            "phi_rad_g": 0.333,
        }

        phi_ig_base = {
            "phi_ig_s": config["phi_ig_s"],
            "phi_ig_f": config["phi_ig_f"],
            "phi_ig_g": config["phi_ig_g"],
            # Optimizing this load has caused unrealistic high temperatures
            # (like 300°C) in basement.
            "phi_ig_b": 0,
        }

        return variables, phi_fraction, phi_ig_base

    @staticmethod
    def evaluate_conifg(
        config,
        obs_all,
        actions_all,
        disturbances_all,
        use_kalman_filter=True,
        return_losses=False,
    ):
        """
        This is a worker function that evaluates the performance of a
        model configuration.
        """
        # Allows using the state estimation methods while this should
        # remain a static mehtod. Fair enough, it would be nicer to
        # have this data as arguments to the state estimation methods.
        # You may want to refactor this sometime.
        bl = Baseline1()
        bl.obs_all = obs_all
        bl.actions_all = actions_all
        bl.disturbances_all = disturbances_all

        variables, phi_fraction, phi_ig_base = bl.tm_args_from_config(config)

        if use_kalman_filter:
            initial_ts = obs_all.index[7]  # Allows 6 steps for Kalman filter.
            initial_state = bl.estimate_initial_state_kalman_filter()
        else:
            initial_ts = obs_all.index[0]
            initial_state = bl.estimate_initial_state_extrapolate_obs()

        tm = ThermalModel(
            weather_data=disturbances_all.loc[initial_ts:],
            variables=variables,
            initial_state=initial_state,
            phi_fraction=phi_fraction,
            phi_ig_base=phi_ig_base,
        )
        tm.reset()
        actions = actions_all.loc[initial_ts:]
        actions_tm = pd.DataFrame(
            index=actions.index,
            data={
                "phi_cca": actions["P_cool_cca"] * 1000,
                "phi_rad": np.zeros(len(actions)) * 1000,
            },
        )
        states_and_dones = tm.step_many(actions=actions_tm)

        residuals = states_and_dones - obs_all - 273.15
        # Remove all columns apart from the zone temperatures
        # (they are filled with NaNs).
        residuals = residuals.dropna(how="all", axis=1).dropna(
            how="all", axis=0
        )
        mae_loss = np.abs(residuals.values).mean()
        rmse_loss = np.square(residuals.values).mean()

        if return_losses:
            return {"mae_loss": mae_loss, "rmse_loss": rmse_loss}
        else:
            tune.report(mae_loss=mae_loss, rmse_loss=rmse_loss)

    def tune_variables(self):
        """
        Black box optimization of the variables of `ThermalModel`.

        Returns:
        --------
        best_config : dict
            The best values of the variables.
        """

        # For most variables we assume that the optimal value is likely
        # around ~ 70% (after initial experiments with 30% deviation showed
        # that some parameters hit the limit) distance to the value choosen
        # by Zwickel et al. We deviate for the following variables:
        # - `A_roof`, `A_wall_gfs` and `A_win_gfs`: Keep fixed as the geometry
        #   of the building is rather certain.
        # - All `C_` variables, as those work against the `f_` and `G_`
        #   variables (chaning one can be compensated by changing the
        #   other too), and we assume that it was likely easier to compute
        #   the thermal capacity then the thermal resistance or the factors.
        # - `eta_cca` and `eta_rad`: Keep fixed as we do not wish to model
        #   to change the eta values to account for other variables, as
        #   changing eta above one makes no sense from a physical point
        #   of view.
        # - `f_sol_roof`, `f_sol_win_gfs` and `f_sol_wall_gfs`:
        #   Are likely much off. Search anything between near zero to
        #   multiples of the original values.
        lower_p = 0.3
        upper_p = 1.7
        search_space = {
            # These are for ThermalModel.variables
            "A_roof": tune.choice(
                [
                    1081.6,
                ]
            ),
            "A_wall_gfs": tune.choice(
                [
                    173.56,
                ]
            ),
            "A_win_gfs": tune.choice(
                [
                    404.96,
                ]
            ),
            "C_roof": tune.uniform(
                lower=3.7121e8 * lower_p, upper=3.7121e8 * upper_p
            ),
            "C_cca": tune.uniform(
                lower=6.9333e8 * lower_p, upper=6.9333e8 * upper_p
            ),
            "C_zone_gfs": tune.uniform(
                lower=1.62246e7 * lower_p, upper=1.62246e7 * upper_p
            ),
            "C_zone_b": tune.uniform(
                lower=5.1146e6 * lower_p, upper=5.1146e6 * upper_p
            ),
            "C_wall_gfs": tune.uniform(
                lower=8.6986e7 * lower_p, upper=8.6986e7 * upper_p
            ),
            "C_wall_b": tune.uniform(
                lower=1.1809e8 * lower_p, upper=1.1809e8 * upper_p
            ),
            "C_floor_fs": tune.uniform(
                lower=3.1475e7 * lower_p, upper=3.1475e7 * upper_p
            ),
            "C_floor_g": tune.uniform(
                lower=1.15162e9 * lower_p, upper=1.15162e9 * upper_p
            ),
            "C_floor_b": tune.uniform(
                lower=3.63038e8 * lower_p, upper=3.63038e8 * upper_p
            ),
            "G_roof_up": tune.uniform(
                lower=888.94 * lower_p, upper=888.94 * upper_p
            ),
            "G_roof_dn": tune.uniform(
                lower=371.60 * lower_p, upper=371.60 * upper_p
            ),
            "G_cca_up": tune.uniform(
                lower=7996.2 * lower_p, upper=7996.2 * upper_p
            ),
            "G_cca_dn": tune.uniform(
                lower=7996.2 * lower_p, upper=7996.2 * upper_p
            ),
            "G_wall_gfs_in": tune.uniform(
                lower=810.03 * lower_p, upper=810.03 * upper_p
            ),
            "G_wall_gfs_out": tune.uniform(
                lower=42.38 * lower_p, upper=42.38 * upper_p
            ),
            "G_wall_b_in": tune.uniform(
                lower=1099.69 * lower_p, upper=1099.69 * upper_p
            ),
            "G_wall_b_out": tune.uniform(
                lower=57.53 * lower_p, upper=57.53 * upper_p
            ),
            "G_window_fs": tune.uniform(
                lower=566.94 * lower_p, upper=566.94 * upper_p
            ),
            "G_floor_fs_up": tune.uniform(
                lower=1054.44 * lower_p, upper=1054.44 * upper_p
            ),
            "G_floor_fs_dn": tune.uniform(
                lower=6216.69 * lower_p, upper=6216.69 * upper_p
            ),
            "G_floor_g_up": tune.uniform(
                lower=409.47 * lower_p, upper=409.47 * upper_p
            ),
            "G_floor_g_dn": tune.uniform(
                lower=1307.96 * lower_p, upper=1307.96 * upper_p
            ),
            "G_floor_b_up": tune.uniform(
                lower=129.08 * lower_p, upper=129.08 * upper_p
            ),
            "G_floor_b_dn": tune.uniform(
                lower=412.32 * lower_p, upper=412.32 * upper_p
            ),
            "eta_cca": tune.choice(
                [
                    1.0,
                ]
            ),
            "eta_rad": tune.choice(
                [
                    1.0,
                ]
            ),
            "f_sol_roof": tune.uniform(lower=1e-5, upper=0.2),
            "f_sol_win_gfs": tune.uniform(lower=1e-5, upper=0.4),
            "f_sol_wall_gfs": tune.uniform(lower=1e-5, upper=0.2),
            "f_vent_gfs": tune.uniform(
                lower=630.957 * lower_p, upper=630.957 * upper_p
            ),
            "f_vent_b": tune.uniform(
                lower=198.903 * lower_p, upper=198.903 * upper_p
            ),
            "f_ground": tune.uniform(
                lower=0.68475 * lower_p, upper=0.68475 * upper_p
            ),
            "f_b": tune.uniform(
                lower=0.31524 * lower_p, upper=0.31524 * upper_p
            ),
            # These are for the ThermalModel.phi_fraction
            "frac_phi_cca_s": tune.uniform(lower=0.2, upper=0.5),
            "frac_phi_cca_f": tune.uniform(lower=0.2, upper=0.5),
            # These are the base loads of the internal gains.
            "phi_ig_s": tune.loguniform(lower=1.0, upper=1e6),
            "phi_ig_f": tune.loguniform(lower=1.0, upper=1e6),
            "phi_ig_g": tune.loguniform(lower=1.0, upper=1e6),
        }

        bbo_search_alg = HEBOSearch(
            metric="mae_loss", mode="min", max_concurrent=40
        )

        trainable = tune.with_parameters(
            self.evaluate_conifg,
            obs_all=self.obs_all,
            actions_all=self.actions_all,
            disturbances_all=self.disturbances_all,
        )

        bbo_results = tune.run(
            trainable,
            metric="mae_loss",
            mode="min",
            config=search_space,
            search_alg=bbo_search_alg,
            num_samples=1000,
            #    checkpoint_freq=0,
            #    keep_checkpoints_num=0,
            resources_per_trial={"cpu": 1},
            verbose=1,
        )

        best_config = bbo_results.best_config

        return best_config
