from setuptools import setup

with open("README", 'r') as f:
    long_description = f.read()

setup(
    name='foo',
    version='1.0',
    description='A useful module',
    license="MIT",
    long_description=long_description,
    author='Man Foo',
    author_email='foomail@foo.com',
    url="http://www.foopackage.com/",
    packages=['foo'],  # same as name
    # external packages as dependencies
    install_requires=['wheel', 'bar', 'greek'],
    scripts=[
        'scripts/cool',
        'scripts/skype',
    ]
)
