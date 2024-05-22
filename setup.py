from setuptools import find_packages, setup

EXCLUDE_FROM_PACKAGES = ["tests", "notebooks", "ai_trader"]
packages = find_packages(exclude=EXCLUDE_FROM_PACKAGES)

setup(
    name="hive_rl_simulator",
    packages=packages,
    include_package_data=True,
    use_scm_version=True,
    setup_requires=["setuptools_scm"],
    description=(
        "HIVE board game RL simulator"
    ),
    author="ali_really_express",
    license="private",
)
