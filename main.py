#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import jax.numpy as jnp
from jax import grad, jit, vmap, random, jacrev, jacfwd
import matplotlib.pyplot as plt
from copy import deepcopy
import numpy as np
from numpy import pi, cos, sin
from typing import Sequence
from functools import partial

class EmbeddedManifold(object):
    def __init__(self):
        pass

    def f(self, xs: Sequence[float]):
        raise NotImplementedError("class EmbeddedManifold is abstract")

    @partial(jit, static_argnums=(0,))
    def df(self, xs: Sequence[float]):
        return jnp.array(jacrev(self.f)(xs))

    @partial(jit, static_argnums=(0,))
    def pushfwd(self, xs: Sequence[float], vs: Sequence[float]):
        df = self.df(xs)
        return jnp.dot(df.transpose(), jnp.array(vs))

    @partial(jit, static_argnums=(0,))
    def g(self, xs: Sequence[float]):
        df = self.df(xs)
        return jnp.dot(df, df.transpose())

    @partial(jit, static_argnums=(0,))
    def dg(self, xs: Sequence[float]):
        # dg[i][j][k] = ∂g_ij / ∂x_k
        return jnp.array(jacrev(self.g)(xs)).transpose(1, 2, 0)

    @partial(jit, static_argnums=(0,))
    def Gamma(self, xs: Sequence[float]):
        # 2nd Christoffel symbol
        # ref: https://manabitimes.jp/physics/1782
        ginv = jnp.linalg.inv(self.g(xs))
        Gamma_aux = self.Gamma_aux(xs)
        return jnp.einsum("ab,bcd->acd", ginv, Gamma_aux)

    @partial(jit, static_argnums=(0,))
    def Gamma_aux(self, xs: Sequence[float]):
        dg = self.dg(xs)
        return jnp.array(0.5 * (dg.transpose(0, 2, 1) + dg.transpose(1, 2, 0) - dg))

    def geodesics(self, x0: Sequence[float], v0: Sequence[float], T=1.0, dt=0.005, embed=False):
        # return the list of position and velocity in chart or embedded R^n
        x, v = deepcopy(x0), deepcopy(v0)
        xs, vs = [], []
        t = 0.0
        while t <= T:
            if embed:
                x_ = self.f(x)
                v_ = self.pushfwd(x, v)
                xs.append(x_)
                vs.append(v_)
            else:
                xs.append(deepcopy(x))
                vs.append(deepcopy(v))

            Gamma = self.Gamma(x)
            acc = -jnp.einsum("abc,b,c->a", Gamma, v, v)
            for i in range(len(acc)):
                v[i] += acc[i] * dt
                x[i] += v[i] * dt

            t += dt

        return xs, vs

class S2(EmbeddedManifold):
    def __init__(self):
        pass

    def f(self, xs: Sequence[float]):
        x, y = xs[0], xs[1]
        denom = 1.0 + x * x + y * y
        return jnp.array([2.0*x, 2.0*y, (-1.0 + x*x + y*y)]) / denom

def main1():
    s2 = S2()
    # case 1
    x0 = [1.0, 2.0]
    v0 = [-1.0, -5.0]
    # case 2
    x0 = [0.02, 0.02]
    v0 = [-1.0, 0.0]
    xs, vs = s2.geodesics(x0, v0, T=1.5, embed=True)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    # ax.set_aspect("equal")
    x, y, z = [], [], []
    v_plot_gain = 1.0 / 6.0
    for i in range(len(xs)):
        p = xs[i]
        x.append(p[0])
        y.append(p[1])
        z.append(p[2])
        if i % 50 == 0:
            v = jnp.array(vs[i])
            v = v / (jnp.linalg.norm(v)) * v_plot_gain
            ax.quiver(p[0], p[1], p[2], v[0], v[1], v[2], color='b', linewidth=2)

    ax.plot3D(x, y, z, color='r', linewidth=2)

    phi, theta = np.mgrid[0.0:pi:20j, 0.0:2.0*pi:20j]
    x = 1.0 * sin(phi) * cos(theta)
    y = 1.0 * sin(phi) * sin(theta)
    z = 1.0 * cos(phi)
    ax.plot_surface(x, y, z, rstride=1, cstride=1, color='gray', alpha=0.4, linewidth=0)
    plt.show()


def main2():
    # plot tangent vector along a curve
    # γ: [0, 1] -> (t^2, -sin(t))
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    t, x, y = [], [], []
    vx, vy = [], []
    px, py, pz = [], [], []
    pvx, pvy, pvz = [], [], []
    v_plot_gain = 1.0 / 6.0
    cnt = 0
    for t_ in np.linspace(0, 1, num=1000):
        cnt += 1
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
        if cnt % 100 == 0:
            ax.quiver(*p, *pv, color='b', linewidth=2)

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


def main3():
    # pararelll transport of a tangent vector
    # https://manabitimes.jp/physics/1783
    # plot tangent vector along a curve
    # γ: [0, 1] -> (t^2, -sin(t))
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    t, x, y = [], [], []
    vx, vy = [], []
    px, py, pz = [], [], []
    pvx, pvy, pvz = [], [], []
    vx1, vy1 = [], []
    # v_origin = jnp.array([2*0.0, -math.cos(0.0)])
    v_origin = jnp.array([-0.5, -0.5])
    vx1_, vy1_ = v_origin[0], v_origin[1]
    dv1 = [0.0, 0.0, 0.0]
    v_plot_gain = 1.0 / 7.5
    v_plot_gain2 = 1.0 / 9.0 / jnp.linalg.norm(v_origin)
    cnt = 0
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
        ## tangent vector along the curve
        p = F([x_, y_])
        px.append(p[0])
        py.append(p[1])
        pz.append(p[2])
        pv = jnp.dot(jnp.array(dF([x_, y_])).transpose(), jnp.array([vx_, vy_]))
        pv = pv / jnp.linalg.norm(pv) * v_plot_gain
        pvx.append(pv[0])
        pvy.append(pv[1])
        pvz.append(pv[2])
        if cnt % 100 == 0:
            ax.quiver(*p, *pv, color='b', linewidth=2)

        ## pararell transport
        vx1_ += dv1[0] * 0.001 # HACK dt = 0.001
        vy1_ += dv1[1] * 0.001
        vx1.append(vx1_)
        vy1.append(vy1_)
        Gamma = christoffel2([x_, y_])
        dv1 = -jnp.einsum("b,abc,c->a", jnp.array([vx1_, vy1_]), Gamma, jnp.array([vx_, vy_]))
        if cnt % 100 == 0:
            v2 = jnp.dot(jnp.array(dF([x_, y_])).transpose(), jnp.array([vx1_, vy1_]))
            v2 = v2 * v_plot_gain2
            ax.quiver(*p, v2[0], v2[1], v2[2], color='g', linewidth=2)
        cnt += 1

    ax.plot3D(px, py, pz, color='r', linewidth=2)
    r = 1
    pi = np.pi
    cos = np.cos
    sin = np.sin
    phi, theta = np.mgrid[0.0:pi:50j, 0.0:2.0 * pi:50j]
    x = r * sin(phi) * cos(theta)
    y = r * sin(phi) * sin(theta)
    z = r * cos(phi)
    ax.plot_surface(
        x, y, z, rstride=1, cstride=1, color='gray', alpha=0.4, linewidth=0)
    plt.show()


if __name__ == '__main__':
    main1()
    #main2()
    #main3()
