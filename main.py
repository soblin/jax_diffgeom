#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import jax.numpy as jnp
from jax import grad, jit, vmap, random, jacrev, jacfwd
import matplotlib.pyplot as plt
from copy import deepcopy
import numpy as np
import math

@jit
def F(params):
    x = params[0]
    y = params[1]
    denom = 1.0 + x * x + y * y
    return jnp.array([2.0*x, 2.0*y, (-1.0 + x*x + y*y)]) / denom

@jit
def dF(params):
    return jnp.array(jacrev(F)(params))

@jit
def g(params):
    df = dF(params)
    return jnp.dot(df, df.transpose())

@jit
def dg(params):
    return jnp.array(jacrev(g)(params)).transpose(1, 2, 0)

def christoffel1(params):
    """
    first christoffel symbol
    https://manabitimes.jp/physics/1782
    """
    G = dg(params)
    return jnp.array(0.5 * (G.transpose(0, 2, 1) + G.transpose(1, 2, 0) - G))

def christoffel2(params):
    G = g(params)
    Ginv = jnp.linalg.inv(G)
    Gamma = christoffel1(params)
    return jnp.einsum("ab,bcd->acd", Ginv, Gamma)

def dF_manual(params):
    x = params[0]
    y = params[1]
    denom = 1.0 + x * x + y * y
    return jnp.array([[2-2*x*x+2*y*y, -4*x*y, 4*x],
                    [-4*x*y, 2+2*x*x-2*y*y, 4*y]]) / (denom*denom)

def g_manual(params):
    df = dF_manual(params)
    return jnp.dot(df, df.transpose())

def geodesic_trajectory(x0: [float], v0: [float], T=1.0, dt=0.005):
    x, v = deepcopy(x0), deepcopy(v0)
    xs, vs = [deepcopy(x)], [deepcopy(v)]
    t = 0.0
    while t < T:
        t += dt
        Gamma = christoffel2(x)
        a = -jnp.einsum("abc,b,c->a", Gamma, v, v)# dv/dt or d^2x/dt^2
        for i in range(len(v)):
            v[i] += a[i] * dt
            x[i] += v[i] * dt
        xs.append(deepcopy(x))
        vs.append(deepcopy(v))
    return xs, vs

def main1():
    params = [1.0, -2.0]
    """
    print(f'F = \n{F(params)}')
    print(f'diff of dF = \n{dF(params) - dF_manual(params)}')
    print(f'diff of g = \n{g(params) - g_manual(params)}')
    print(f'Γ1 = \n{christoffel1(params)}')
    print(f'Γ2 = \n{christoffel2(params)}')
    """
    # case 1
    x0 = [1.0, 2.0]
    v0 = [-1.0, -5.0]
    # case 2
    x0 = [0.02, 0.02]
    v0 = [-1.0, 0.0]
    traj = geodesic_trajectory(x0, v0, 5.0)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    # ax.set_aspect("equal")
    xs, ys, zs = [], [], []
    v_plot_gain = 1.0 / 6.0
    for i in range(len(traj[1])):
        param = traj[1][i]
        p = F(param)
        xs.append(p[0])
        ys.append(p[1])
        zs.append(p[2])
        if i % 100 == 0:
            param_v = jnp.array(traj[1][i])
            # df(param) is the 'differential' of F that maps TM1 to TM2
            v = jnp.dot(jnp.array(dF(param)).transpose(), param_v)
            #print(v)
            v = v / (jnp.linalg.norm(v)) * v_plot_gain
            ax.quiver(*p, *v, color='b', linewidth=2)
    ax.plot3D(xs, ys, zs, color='r', linewidth=2)

    """
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = np.cos(u)*np.sin(v)
    y = np.sin(u)*np.sin(v)
    z = np.cos(v)
    ax.plot_wireframe(x, y, z, color="r")
    """
    r = 1
    pi = np.pi
    cos = np.cos
    sin = np.sin
    phi, theta = np.mgrid[0.0:pi:100j, 0.0:2.0 * pi:100j]
    x = r * sin(phi) * cos(theta)
    y = r * sin(phi) * sin(theta)
    z = r * cos(phi)
    ax.plot_surface(
        x, y, z, rstride=1, cstride=1, color='gray', alpha=0.4, linewidth=0)
    plt.show()


def main2():
    # γ: [0, 1] -> (t^2, -sin(t))
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    t, x, y = [], [], []
    vx, vy = [], []
    px, py, pz = [], [], []
    pvx, pvy, pvz = [], [], []
    v_plot_gain = 1.0 / 6.0
    for t_ in np.linspace(0, 1, num=1000):
        # in chart
        t.append(t_)
        x_ = math.pow(t_, 2)
        y_ = -math.sin(t_)
        vx_ = 2 * t_
        vy_ = -math.cos(t_)
        x.append(x_)
        y.append(y_)
        vx.append(vx_)
        vy.append(vy_)

        # in embedded manifold
        p = F([x_, y_])
        px.append(p[0])
        py.append(p[1])
        pz.append(p[2])
        pv = jnp.dot(jnp.array(dF([x_, y_])).transpose(), jnp.array([vx_, vy_]))
        pv = pv / jnp.linalg.norm(pv) * v_plot_gain
        pvx.append(pv[0])
        pvy.append(pv[1])
        pvz.append(pv[2])

    ax.plot3D(px, py, pz, color='r', linewidth=2)
    r = 1
    pi = np.pi
    cos = np.cos
    sin = np.sin
    phi, theta = np.mgrid[0.0:pi:100j, 0.0:2.0 * pi:100j]
    x = r * sin(phi) * cos(theta)
    y = r * sin(phi) * sin(theta)
    z = r * cos(phi)
    ax.plot_surface(
        x, y, z, rstride=1, cstride=1, color='gray', alpha=0.4, linewidth=0)
    plt.show()


if __name__ == '__main__':
    #main1()
    main2()
