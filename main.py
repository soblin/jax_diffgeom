#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import jax.numpy as jnp
from jax import grad, jit, vmap, random, jacrev, jacfwd
import matplotlib.pyplot as plt
from copy import deepcopy
import numpy as np
from numpy import pi, cos, sin
import math
from typing import Sequence
from functools import partial


class EmbeddedManifold(object):
    def __init__(self):
        pass

    def f(self, xs: Sequence[float]):
        raise NotImplementedError("class EmbeddedManifold is abstract")

    # workaround for @jit of member function
    # ref: https://github.com/google/jax/issues/1251
    @partial(jit, static_argnums=(0,))
    def df(self, xs: Sequence[float]):
        return jnp.array(jacfwd(self.f)(xs))

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
        # dg[i][j][k] := ∂g_ij / ∂x_k
        return jnp.array(jacfwd(self.g)(xs))

    @partial(jit, static_argnums=(0,))
    def Gamma(self, xs: Sequence[float]):
        # 2nd Christoffel symbol
        # ref: https://manabitimes.jp/physics/1782
        Dg = self.dg(xs)
        ginv = jnp.linalg.inv(self.g(xs))
        return 0.5 * (jnp.einsum("im,mkl->ikl", ginv, Dg)
                      + jnp.einsum("im,mlk->ikl", ginv, Dg)
                      - jnp.einsum("im,klm->ikl", ginv, Dg))

    def accel(self, xs: Sequence[float],  vs: Sequence[float]):
        Gamma = self.Gamma(xs)
        v = jnp.array(vs)
        return -jnp.einsum("abc,b,c->a", Gamma, v, v)

    #@partial(jit, static_argnums=(0,))
    def geodesics(self, x0: Sequence[float], v0: Sequence[float], T: float, dt: float):
        # return the list of position and velocity in chart or embedded R^n
        x, v = deepcopy(x0), deepcopy(v0)
        xs, vs = [], []
        t = 0.0
        while t <= T:
            x_ = self.f(x)
            v_ = self.pushfwd(x, v)
            xs.append(x_)
            vs.append(v_)

            acc = self.accel(x, v)
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
        denom = 1.0 + x*x + y*y
        return jnp.array([2.0*x, 2.0*y, (1.0 - x*x - y*y)]) / denom


def main1():
    s2 = S2()
    # case 1
    x0 = [1.0, 2.0]
    v0 = [-1.0, -5.0]
    # case 2
    x0 = [0.0, 0.0]
    v0 = [1.0, -1.0]
    xs, vs = s2.geodesics(x0, v0, 1.5, 0.005)

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
            ax.quiver(p[0], p[1], p[2], v[0], v[1],
                      v[2], color='b', linewidth=2)

    ax.plot3D(x, y, z, color='r', linewidth=2)

    phi, theta = np.mgrid[0.0:pi:20j, 0.0:2.0*pi:20j]
    x = 1.0 * sin(phi) * cos(theta)
    y = 1.0 * sin(phi) * sin(theta)
    z = 1.0 * cos(phi)
    ax.plot_surface(x, y, z, rstride=1, cstride=1,
                    color='gray', alpha=0.4, linewidth=0)
    plt.show()


def main2():
    s2 = S2()
    # plot tangent vector along a curve γ: [0, 1] -> (t^2, -sin(t))
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    # underbar: in chart
    xt_, yt_ = [], []
    vtx_, vty_ = [], []
    # in R^3
    xt, yt, zt = [], [], []
    vtx, vty, vtz = [], [], []
    v_plot_gain = 1.0 / 8.0
    cnt = 0
    for t in np.linspace(0, 1, num=1001):
        # in chart
        xt_ = math.pow(t, 2)
        yt_ = -math.sin(t)
        vtx_ = 2 * t
        vty_ = -math.cos(t)

        # in embedded manifold
        p = s2.f([xt_, yt_])
        for l_, v_ in zip([xt, yt, zt], [p[0], p[1], p[2]]):
            l_.append(v_)
        v = s2.pushfwd([xt_, yt_], [vtx_, vty_])
        v = v / jnp.linalg.norm(v) * v_plot_gain
        for l_, v_ in zip([vtx, vty, vtz], [v[0], v[1], v[2]]):
            l_.append(v_)
        if cnt % 100 == 0:
            ax.quiver(p[0], p[1], p[2], v[0], v[1],
                      v[2], color='b', linewidth=2)

        cnt += 1

    ax.plot3D(xt, yt, zt, color='r', linewidth=2)
    phi, theta = np.mgrid[0.0:pi:20j, 0.0:2.0 * pi:20j]
    x = 1.0 * sin(phi) * cos(theta)
    y = 1.0 * sin(phi) * sin(theta)
    z = 1.0 * cos(phi)
    ax.plot_surface(x, y, z, rstride=1, cstride=1,
                    color='gray', alpha=0.4, linewidth=0)
    plt.show()


def main3():
    # plot tangent vector along a curve γ: [0, 1] -> (t^2, -sin(t))
    # ref: https://manabitimes.jp/physics/1783
    s2 = S2()
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    # underbar: in chart
    gamma_xts, gamma_yts = [], []
    # in R^3
    xts, yts, zts = [], [], []
    vxts, vyts, vzts = [], [], []
    v_plot_gain = 1.0 / 8.0
    # vector to be transported
    v = [-0.5, -0.5]  # in chart
    dv = [0.0, 0.0]  # in chart
    norms = []
    Jvs = []
    cnt = 0
    num = 1001;
    for t in np.linspace(0, 1, num=num):
        # in chart
        gamma_xt = math.pow(t, 2)
        gamma_yt= -math.sin(t)
        gamma_vxt = 2*t
        gamma_vyt = -math.cos(t)
        gamma_Lvt = math.hypot(gamma_vxt, gamma_vyt)
        # gamma_vxt /= gamma_Lvt
        # gamma_vyt /= gamma_Lvt
        # in embedded manifold
        [xt, yt, zt] = s2.f([gamma_xt, gamma_yt])
        for l_, v_ in zip([xts, yts, zts], [xt, yt, zt]):
            l_.append(v_)
        [vxt, vyt, vzt] = s2.pushfwd([gamma_xt, gamma_yt], [gamma_vxt, gamma_vyt])
        for l_, v_ in zip([vxts, vyts, vzts], [vxt, vyt, vzt]):
            l_.append(v_)
        
        # parallel transport
        [Jvxt, Jvyt, Jvzt] = s2.pushfwd([gamma_xt, gamma_yt], v)  # in R^3
        Jvs.append([Jvxt, Jvyt, Jvzt])
        Gamma = s2.Gamma([gamma_xt, gamma_yt])

        # plot
        if cnt % 100 == 0:
            Jvxt_plt = Jvxt * v_plot_gain;
            Jvyt_plt = Jvyt * v_plot_gain;
            Jvzt_plt = Jvzt * v_plot_gain;
            ax.quiver(xt, yt, zt, Jvxt_plt, Jvyt_plt, Jvzt_plt, color='b', linewidth=2)
            Lvt = math.sqrt(vxt*vxt + vyt*vyt + vzt*vzt)
            vxt_plt = vxt / Lvt * v_plot_gain
            vyt_plt = vyt / Lvt * v_plot_gain
            vzt_plt = vzt / Lvt * v_plot_gain
            ax.quiver(xt, yt, zt, vxt_plt, vyt_plt, vzt_plt, color='g', linewidth=3)
            g = s2.g([gamma_xt, gamma_yt])
            v_ = jnp.array(v)
            norm = math.sqrt(v_.transpose().dot(g).dot(v_))
            norms.append(norm)
            print(f"norm: in chart: {norm}: in R^3: ", math.sqrt(Jvxt * Jvxt + Jvyt * Jvyt + Jvzt * Jvzt))
        dv = -jnp.einsum("abc,b,c->a", Gamma, [gamma_vxt, gamma_vyt], v)
        for i in range(len(dv)):
            v[i] += dv[i] * (1.0 / num)

        cnt += 1

    ax.plot3D(xt, yt, zt, color='r', linewidth=2)
    phi, theta = np.mgrid[0.0:pi:20j, 0.0:2.0 * pi:20j]
    x = 1.0 * sin(phi) * cos(theta)
    y = 1.0 * sin(phi) * sin(theta)
    z = 1.0 * cos(phi)
    ax.plot_surface(x, y, z, rstride=1, cstride=1,
                    color='gray', alpha=0.4, linewidth=0)
    plt.show()
    plt.plot(norms)
    plt.show()


def check_jac_dimension():
    def f(x):
        return jnp.array([x[0], x[1]])

    def df(x):
        return jnp.array(jacrev(f)(x))

    def g(x):
        return jnp.array([[jnp.linalg.norm(x), 0], [0, jnp.linalg.norm(x)]])

    def dg(x):
        return jnp.array(jacrev(g)(x))

    x = [1.0, 2.0, 3.0]
    ret = df(x)
    print(ret.shape)  # -> (3,2)
    #ret = ret.transpose(1, 2, 0)
    ret = dg(x)
    print(ret.shape)  # -> (3,2,2)
    ret = ret.transpose(1, 2, 0)
    print(ret.shape)  # -> (2,2,3)


if __name__ == '__main__':
    # main1()
    # main2()
    main3()
    # check_jac_dimension()
