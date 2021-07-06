from __future__ import print_function
import os
from datetime import date
from setuptools import setup, find_packages
# --- import your package ---

directory = os.path.dirname(os.path.abspath(__file__))

VERSION = '0.1.3'
SHORT_DESCRIPTION = "NLP error analysis."
LICENSE = "BSD 3-Clause License"
AUTHOR = "Tongshuang Wu"
AUTHOR_EMAIL = "wtshuang@cs.washington.edu"
MAINTAINER = "Tongshuang Wu"
MAINTAINER_EMAIL = "wtshuang@cs.washington.edu"
GITHUB_USERNAME = "tongshuangwu"
PKG_NAME = "polyjuice_nlp"

if __name__ == "__main__":
    # --- Automatically generate setup parameters ---
    # Your package name

    PACKAGES, INCLUDE_PACKAGE_DATA, PACKAGE_DATA, PY_MODULES = (
        None, None, None, None,
    )

    # It's a directory style package
    if os.path.exists(__file__[:-8] + PKG_NAME):
        # Include all sub packages in package directory
        PACKAGES = [PKG_NAME] + ["%s.%s" % (PKG_NAME, i)
                                 for i in find_packages(PKG_NAME)]

        # Include everything in package directory
        INCLUDE_PACKAGE_DATA = True
        PACKAGE_DATA = {
            "": ["*.*"],
        }

    # It's a single script style package
    elif os.path.exists(__file__[:-8] + PKG_NAME + ".py"):
        PY_MODULES = [PKG_NAME, ]

    # The project directory name is the GitHub repository name
    repository_name = os.path.basename(os.path.dirname(__file__))

    # Project Url
    URL = "https://github.com/{0}/{1}".format(GITHUB_USERNAME, repository_name)
    # Use todays date as GitHub release tag
    github_release_tag = str(date.today())

    PLATFORMS = [
        "Windows",
        "MacOS",
        "Unix",
    ]

    CLASSIFIERS = [
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
        "Operating System :: Unix",
        "Programming Language :: Python",
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ]
    # Long description will be the body of content on PyPI page
    try:
        path = os.path.join(directory, 'README.md')
        with open(path) as read_file:
            LONG_DESCRIPTION = read_file.read()
    except:
        LONG_DESCRIPTION = "No long description!"
    # Read requirements.txt, ignore comments
    try:
        REQUIRES = list()
        f = open("requirements.txt", "rb")
        for line in f.read().decode("utf-8").split("\n"):
            line = line.strip()
            if "#" in line:
                line = line[:line.find("#")].strip()
            if line:
                REQUIRES.append(line)
    except:
        print("'requirements.txt' not found!")
        REQUIRES = list()

    setup(
        name=PKG_NAME,
        description=SHORT_DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        long_description_content_type="text/markdown",
        version=VERSION,
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        packages=PACKAGES,
        include_package_data=INCLUDE_PACKAGE_DATA,
        package_data=PACKAGE_DATA,
        py_modules=PY_MODULES,
        url=URL,
        classifiers=CLASSIFIERS,
        platforms=PLATFORMS,
        license=LICENSE,
        install_requires=REQUIRES,
    )