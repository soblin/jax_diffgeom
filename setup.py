# -*- coding: utf-8 -*-
#!/usr/bin/env python3

setup(
    name="diffgeom_jax",
    version="1.0",
    description="Differential geometry using JAX",
    author="soblin",
    packages=['diffgeom_jax'],
    python_requires=">=3.6",
    install_requires=["numpy", "scipy", "matplotlib", "jax", "jaxlib"],
)
