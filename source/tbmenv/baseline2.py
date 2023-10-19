#!/usr/bin/env python3
"""
"""
from datetime import time
from functools import cache

import numpy as np
import pandas as pd
from scipy.optimize import least_squares


class Baseline2:
    """
    Baseline for Scenario 2 using reusing the building model from here:
    https://github.com/fzi-forschungszentrum-informatik/tropical_precooling_environment/blob/master/tropical_precooling/env.py
    """

    def __init__(self):
        self.obs_all = {}
        self.actions_all = {}
        self.disturbances_all = {}

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
        if not isinstance(obs, list):
            obs = [obs]
        if not isinstance(actions, list):
            actions = [actions]
        if not isinstance(disturbances, list):
            disturbances = [disturbances]

        for o in obs:
            self.obs_all[o.index[0].date()] = o
        for a in actions:
            self.actions_all[a.index[0].date()] = a
        for d in disturbances:
            self.disturbances_all[d.index[0].date()] = d

    @staticmethod
    def simulate_day(bparams, T_z_0, actions, disturbances):
        """
        Simulate the zone temperature for one day.

        This starts with the temperature measured at the real building at
        4:57:30am and computes change of temperatue within the time step
        length of 5 minutes, by applying equations (3), (4) and (5) from
        the paper. This is repeated until the full day horizon is simulated.

        Parameters
        ----------
        bparams : numpy.Array
            The 5 variables of the building model.
        T_z_0 : float
            Zone temperature at 4:57:30am
        actions : pandas.DataFrame
            Temperature setpoints (column: `"T_zSP"`) for every 5 minute slot
            between 5am and 5pm.
        disturbances : pandas.DataFrame
            Predictions of disturbances for every 5 minute slot.
            (columns: '`["T_a", "T_s_fct_mean", "CO2_fct_mean"]`)

        Returns
        -------
        T_z : pandas.DataFrame
            The computed zone temperature of the simulated day.
        """
        k_a = bparams[0]
        k_o1 = bparams[1]
        k_o2 = bparams[2]
        m_so = bparams[3]
        k_c = bparams[4]
        c_pa = 1.005
        C_z = 1.0

        # Our container to store the zone temperature.
        T_z = []

        # Make some arrangements to make the notiations below follow
        # the notation in equations given in the paper.
        sim_data = pd.DataFrame(index=actions.index)
        sim_data["T_zSP_t"] = actions["T_zSP"]
        sim_data["T_s_t"] = disturbances["T_s_fct_mean"]
        sim_data["T_a_t"] = disturbances["T_a"]
        sim_data["theta_CO2_t"] = disturbances["CO2_fct_mean"]

        sim_data = sim_data.loc[sim_data.index.time > time(5, 0, 0)]

        # This is the zone temperature at 04:47:30.
        T_z_t = T_z_0
        # Iterate over rows of sim_data to conveniently get the values
        # for each of the 5 minute blocks.
        for _, row in sim_data.iterrows():
            T_zSP_t = row["T_zSP_t"]
            if T_zSP_t is None:
                # Treat AC off as zone has setpoint.
                T_zSP_t = T_z_t

            T_s_t = row["T_s_t"]
            T_a_t = row["T_a_t"]
            theta_CO2_t = row["theta_CO2_t"]

            # Compute the delta for the zone temperature of the
            # next timestep.
            #
            m_s_t = m_so + k_c * (T_z_t - T_zSP_t)  # (5)
            # Prevent negative energy prices if the agent sets the setpoint
            # above the current temperature. See for dicussion:
            # https://github.com/fzi-forschungszentrum-informatik/tropical_precooling_environment/issues/1
            m_s_t = np.clip(m_s_t, 0, None)
            Q_cooling_t = c_pa * (m_s_t * (T_s_t - T_z_t))  # (4)

            # No cooling/heating if AC is switched of.
            if np.isnan(Q_cooling_t):
                Q_cooling_t = 0

            # (5)
            dT_dt = k_a * (T_a_t - T_z_t)
            dT_dt += k_o1 * theta_CO2_t + k_o2
            dT_dt += Q_cooling_t
            dT_dt /= C_z
            dT = dT_dt * 300  # 5 Minutes step length

            T_z_t = T_z_t + dT

            # Store the current zone temperature.
            T_z.append(T_z_t)

        T_z_predicted = pd.DataFrame(index=sim_data.index, data={"T_z": T_z})

        return T_z_predicted

    @staticmethod
    def _optimize_parameters_worker(
        bparams, sim_fn, sim_fn_kwargs, observed_states
    ):
        """
        A worker function for finding optimized parameters that can be
        evaluated in parallel.
        """
        predicted_states = sim_fn(bparams, **sim_fn_kwargs)
        residiual = abs(observed_states - predicted_states).dropna()
        if len(residiual) != 144:
            raise RuntimeError(
                "Seems like simulating the building model has returned not"
                "the expected data length. There must be a bug in the code."
            )
        return residiual.values.flatten()

    @cache
    def find_optimized_parameters(self, date):
        """
        Finds the optimial parameters for a particular date using least squares.
        """
        # Initial solution should have at least the rough matching magnitude.
        bparams_0 = np.ones(5) * 10**-5

        sim_fn_kwargs = {
            # NOTE: this should actually be the temperature at 4:57:30, but
            # the latter is not part of the obs df available under `date` and
            # we skip the extra date handling here as the temperatures should
            # not diverge much.
            "T_z_0": self.obs_all[date].loc[time(5, 2, 30), "T_z"][0],
            "actions": self.actions_all[date],
            "disturbances": self.disturbances_all[date],
        }
        observed_states = self.obs_all[date]

        opt_result = least_squares(
            fun=self._optimize_parameters_worker,
            x0=bparams_0,
            kwargs={
                "sim_fn": self.simulate_day,
                "sim_fn_kwargs": sim_fn_kwargs,
                "observed_states": observed_states,
            },
        )
        return opt_result.x

    def predict(self, n_dates=1):
        """
        Triggers a prediction for the last day for which data is available.

        This will trigger the optimization of parameters for one or more
        past days to allow for extrapolation of these for the current day.

        Arguments:
        ----------
        n_dates : int
            The number of historic dates that should be used for planning.
        """
        # These are the last 5 days we have obs for. We use these
        # to estimate the optimal coefficients of the model.
        last_obs_dates = sorted(self.obs_all)[-n_dates:]
        bparams_history = np.stack(
            [self.find_optimized_parameters(dt) for dt in last_obs_dates]
        )
        # Compute the average of the parameters.
        bparams_predict = bparams_history.mean(axis=0)

        # Get the last date for which actions exist, this is the date
        # we want to predict.
        action_dates = list(self.actions_all)
        predict_date_i = np.argmax(action_dates)
        predict_date = action_dates[predict_date_i]
        last_obs_date = action_dates[predict_date_i - 1]

        state_prediction = self.simulate_day(
            bparams=bparams_predict,
            T_z_0=self.obs_all[last_obs_date].loc[time(4, 57, 30), "T_z"][0],
            actions=self.actions_all[predict_date],
            disturbances=self.disturbances_all[predict_date],
        )

        return state_prediction
