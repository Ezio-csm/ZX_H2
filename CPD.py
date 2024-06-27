import numpy as np
import pandas as pd
import ruptures as rpt
import statsmodels.api as sm


def std_cpd(w, y=None):
    def is_segment_stable(segment):
        segment += np.random.rand(len(segment)) * 1e-3
        model = sm.OLS(segment[1:], sm.add_constant(segment[:-1])).fit()
        return model.pvalues[1] > 0.95
    algo = rpt.Pelt(model="l1").fit(w)
    result = algo.predict(pen=100)
    stable_change_points = []
    for i, t0 in enumerate(result[:-1]):
        for t in range(t0 + 50, result[i + 1]):
            if is_segment_stable(w[t - 50:t]):
                if y is not None:
                    if is_segment_stable(y[t - 50:t]):
                        stable_change_points.append(t)
                        break
                else:
                    stable_change_points.append(t)
                    break
    return stable_change_points