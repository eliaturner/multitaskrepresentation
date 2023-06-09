import math

import numpy as np
import scipy
from numpy import dot
from numpy.linalg import norm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import svm
import matplotlib.pyplot as plt

def svm_if_separable(points1, points2):
    X = np.vstack((points1, points2))
    X = StandardScaler().fit_transform(X)
    clf = svm.SVC(kernel='linear', C=100, max_iter=10000)
    y = np.ones(len(points1) + len(points2))
    y[len(points1):] = -1
    clf.fit(X, y)
    score = clf.score(X, y)
    # print(score)
    if False:
        pca = PCA(2)
        pca.fit(X)
        plt.clf(); plt.scatter(*np.hsplit(pca.transform(points1), 2)); plt.scatter(*np.hsplit(pca.transform(points2), 2)); plt.show()

    if score == 1:
        width = 2 / np.linalg.norm(clf.coef_)
        return width

    return 0


def get_length(vec):
    return sum([np.linalg.norm(v - u) for u, v in zip(vec[:-1], vec[1:])])


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def calc_angle(vec1, vec2):
    cos_sim = dot(vec1, vec2) / (norm(vec1) * norm(vec2))
    return math.degrees(np.arccos(cos_sim))


def cosine_similarity(v1, v2):
    from scipy import spatial
    return 1 - spatial.distance.cosine(v1, v2)
    return np.abs(1 - spatial.distance.cosine(v1, v2))



def participation_ratio(points):
    points = StandardScaler().fit_transform(points)
    cov = np.cov(points.transpose())
    vals, _ = scipy.linalg.eigh(cov)
    numerator = np.sum(vals)**2
    denominator = np.sum(vals**2)
    return numerator/denominator


def random_noisy_states(init, reps, sigma):
    states = np.tile(init, (reps, 1))
    noisy_states = states + np.random.normal(scale=sigma, size=states.shape)
    return noisy_states

def calc_normalized_q_value(states):
    states_diff = states - np.roll(states, -1, axis=-2)
    q_vals = np.sqrt(np.sum(np.multiply(states_diff, states_diff), axis=-1) / states.shape[-1])
    q_vals[..., -1] = q_vals[..., -2]
    return np.expand_dims(q_vals, axis=-1)

def calc_normalized_q_value_diff(states):
    states_diff = states - np.roll(states, -1, axis=-2)
    q_vals = np.sqrt(np.sum(np.multiply(states_diff, states_diff), axis=-1) / states.shape[-1])
    q_vals[..., -1] = q_vals[..., -2]
    return np.expand_dims(np.abs(np.diff(q_vals)), axis=-1)


def numeric_jacobian(f, x, dx=1e-2):
    n = len(x)
    func = f(x)
    jac = np.zeros((n, n))
    for j in range(n): #through columns to allow for vector addition
        #Dxj = (abs(x[j])*dx if x[j] != 0 else dx)
        x_plus = np.copy(x)
        x_plus[j] += dx
        jac[:, j] = (f(x_plus)-func)/dx
    return jac

from scipy.optimize import minimize

def solve_optimization(A, b):
    # Objective function
    def objective(x):
        return np.linalg.norm(np.dot(A, x) - b)

    # Initial values for the variables to be optimized
    x0 = np.zeros(A.shape[1])
    # Solve the optimization problem
    res = minimize(objective, x0)
    return res.x, res.fun



def curvature_2d(curve):
    dx_dt = np.gradient(curve[:, 0])
    dy_dt = np.gradient(curve[:, 1])
    velocity = np.array([[dx_dt[i], dy_dt[i]] for i in range(dx_dt.size)])
    ds_dt = np.sqrt(dx_dt * dx_dt + dy_dt * dy_dt)
    tangent = np.array([1 / ds_dt] * 2).transpose() * velocity

    tangent_x = tangent[:, 0]
    tangent_y = tangent[:, 1]

    deriv_tangent_x = np.gradient(tangent_x)
    deriv_tangent_y = np.gradient(tangent_y)

    # dT_dt = np.array(
    #     [[deriv_tangent_x[i], deriv_tangent_y[i]] for i in range(deriv_tangent_x.size)])
    #
    # length_dT_dt = np.sqrt(
    #     deriv_tangent_x * deriv_tangent_x + deriv_tangent_y * deriv_tangent_y)

    # normal = np.array([1 / length_dT_dt] * 2).transpose() * dT_dt
    # d2s_dt2 = np.gradient(ds_dt)
    d2x_dt2 = np.gradient(dx_dt)
    d2y_dt2 = np.gradient(dy_dt)

    curvature = np.abs(d2x_dt2 * dy_dt - dx_dt * d2y_dt2) / (
            dx_dt * dx_dt + dy_dt * dy_dt) ** 1.5

    return curvature