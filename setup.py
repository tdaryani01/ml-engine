"""Only marks the wheel as platform-specific: it carries a compiled library (utils/bin/conv_kernels.so or .dll), so a "pure Python" wheel tag would be wrong. Everything else is in pyproject.toml."""
from setuptools import Distribution, setup


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True


setup(distclass=BinaryDistribution)
