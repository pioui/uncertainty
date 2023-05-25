#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup

requirements = ["numpy"]

# setup_requirements = [
#     "pytest-runner",
# ]
# test_requirements = [
#     "pytest",
# ]

setup(
    description="Uncertainty Measurements",
    version="0.1",
    url="https://github.com/pioui/uncertainty",
    author="Pigi Lozou",
    author_email="piyilozou@gmail.com",
    license="MIT",
    install_requires=requirements,
    include_package_data=True,
    keywords="uncertainty",
    name="uncertainty",
    packages=["uncertainty", "uncertainty.datasets"],
    # setup_requires=setup_requirements,
    test_suite="tests",
    # tests_require=test_requirements,
    zip_safe=False,
)

