Development Setup
=================

Note that at the moment we don't accept contributions unless there is an established collaboration. If you are interested to collaborate, feel free to reach out. This is mainly due to the time it takes to coordinate software development in a large group of people, the challenge of developing software openly in academia and licensing concerns. 

## Pre-Commit Config

We are using [pre-commit](https://pre-commit.com) to check commits before they hit code-review. The pre-commit config can be found in root of the repository. At the moment we do the following things

- Format with `black`
- Lint with `flake8`
- Type check with `mypy`
- Synchronise Jupyter notebooks with markdown versions of the same notebooks with `jupytext`.

## Code Review

We are using Gerrit for code review. The server is a [https://gerrit.bioai.eu](https://gerrit.bioai.eu), if you are a new contributor talk to one of the maintainers to get an account and familiarise yourself with the review process that Gerrit imposes.