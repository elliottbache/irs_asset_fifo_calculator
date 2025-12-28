<!-- docs:start -->

# IRS asset FIFO calculator

[![Documentation Status](https://readthedocs.org/projects/irs-asset-fifo-calculator/badge/?version=latest)](https://irs-asset-fifo-calculator.readthedocs.io/en/latest/?badge=latest)
[![CI](https://github.com/elliottbache/irs_asset_fifo_calculator/actions/workflows/ci.yaml/badge.svg?branch=main)](https://github.com/elliottbache/irs_asset_fifo_calculator/actions/workflows/ci.yaml)
[![codecov](https://codecov.io/github/elliottbache/irs_asset_fifo_calculator/graph/badge.svg?token=PFPO48XQ72)](https://codecov.io/github/elliottbache/irs_asset_fifo_calculator) 
[![Release](https://img.shields.io/github/v/release/elliottbache/irs_asset_fifo_calculator)](https://github.com/elliottbache/irs_asset_fifo_calculator/releases)
[![License: GPL-3.0](https://img.shields.io/badge/license-%20%20GNU%20GPLv3%20-green?style=plastic)](https://github.com/elliottbache/irs_asset_fifo_calculator/blob/main/LICENSE)

Tax calculator that tracks capital gains from multiple purchases and sales.  This program uses a CSV file as input.  

This file is called "asset_tx.csv" in the published example, but any name can be
be used, using this name in the python call.  The file has the following header:
"Date", "Asset", "Amount (asset)", "Sell price (\$)", "Buy price (\$)", "Account number", "Entity", "Notes", "Remaining"

**Table of Contents**

- [What this project does](#what-this-project-does)
- [FIFO in one paragraph](#fifo-in-one-paragraph)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Technologies](#technologies)
- [Development](#development)
- [Contributing](#contributing)
- [Contributors](#contributors)
- [Author](#author)
- [Change log](#change-log)
- [License](#license)

## What this project does

This repository implements a small Python tool to calculate IRS-style
capital gains using the FIFO (First In, First Out) method and produce
Form 8949–style output.

Given a CSV of asset transactions (buys, sells, exchanges, transfers),
the library:

1. Groups related rows by `Tx Index` into logical “blocks” (one trade).
2. Parses each block into:
   - **Buy side** (what you acquired)
   - **Sell side** (what you disposed of)
   - **Fees** (possibly in one or more assets)
3. Maintains a **FIFO ledger** of “lots” for each asset (amount, price,
   cost basis, date).
4. For each sale, consumes the oldest lots first to compute:
   - Cost basis
   - Proceeds
   - Gain or loss
5. Writes the result as rows suitable for **Form 8949**.

## FIFO in one paragraph

Under FIFO, the **earliest purchased units are considered sold first**.
If you bought 10 NVDA on January 1 and 5 NVDA on February 1, then sell
12 NVDA on March 1, the sale is treated as:

- 10 units from the January lot, and  
- 2 units from the February lot.

Each slice gets a proportional share of the total proceeds, and its own
cost basis and gain/loss. This tool automates that book-keeping and
emits one Form 8949 row per “slice”.

For a more detailed explanation (with tables and numeric examples),
see [`docs/fifo_overview.md`](docs/fifo_overview.md).

## Quickstart
### Download repo
In an Ubuntu/WSL terminal:
```bash
sudo apt install -y git
git clone https://github.com/elliottbache/irs_asset_fifo_calculator.git
cd irs_asset_fifo_calculator
```

### Quickstart (recommended): Local (Ubuntu/WSL)
In an Ubuntu/WSL terminal:
```bash
make setup
make run
```
That's it, you’ve run the FIFO tax calculator!  Keep reading for a more in-depth explanation
of what just happened.  

#### Tutorial mode and expected results
Optional: compare your results to the expected tutorial results.  If you want a deterministic “known-good” run
you can compare against (useful for demos, onboarding, and quick sanity checks), run the tax calculator
with the example input file and compare the produced Form 8949 to the committed expected file.  
```bash
make tutorial
```
Tutorial example files live here:
- `examples/asset_tx.csv`
- `examples/form8949.csv`

Note: the ```make``` command calls ```bash scripts/compare-tutorial-results.sh```.  This script may loser its
permissions depending on the method of cloning/downloading the repo.  If it does lose its executable
permission, it can be set with
```bash
chmod u+x scripts/compare-tutorial-results.sh
```

### Quickstart (alternative): Docker
Use this if you prefer Docker.  Otherwise, use the [local quickstart](#quickstart-recommended-local-ubuntuwsl) 
 above.
#### Launch Docker daemon
On WSL:
```bash
sudo service docker start
```
On Ubuntu:
```bash
sudo systemctl start docker
```

#### Start a docker container
```bash
docker start <name>
```
#### Then run
```bash
docker compose up --build
```

## Installation (manual, for development or troubleshooting)
If you used [Quickstart (make setup)](#quickstart), you can skip this section.

This package is intended for use in Ubuntu/WSL.  All installation and execution instructions are for these
distributions.  

The quickest and easiest way to install the various components of this package can be found in [Quickstart](#quickstart).
The following steps are for manual installation.
### Create a Python virtual environment with dependencies (skip this if using Docker)
#### System requirements (Ubuntu/WSL):
- **Python**: Python **3.11** + venv support (```python3.11```, ```python3.11-venv```)

#### These are installed via:
- ```make deps``` (runs the scripts below), or
- ```bash scripts/install-python-deps.sh```

Note: if downloaded with wget or as a zip file, the permissions may be lost on the scripts.  In this case,
you may need to change the permissions with ```chmod +x scripts/install-python-deps.sh```.

#### Prefer zero system dependencies?
- Use **Docker** instead (see [Quickstart (alternative): Docker](#quickstart-alternative-docker) below).

#### Create and activate a venv
```bash
python -m venv .venv
. .venv/bin/activate 
```
#### Install rest of dependencies in venv 
```bash
pip install -U pip
pip install -e .[dev]
```

## Execution / Usage (manual, for development or troubleshooting)
If you used [Quickstart (make setup)](#quickstart), you can skip this section.

This program was developed with Python 3.11.14.  It is intended for use in Ubuntu/WSL.

### Option A: No Docker
#### Run demo
From within the Python virtual environment (see [Virtual environment](#create-and-activate-a-venv)):
```bash
irs-fifo-taxes
```
Input file and output file flags are available for running in CLI.  e.g.
```bash
irs-fifo-taxes --input-file="my_special_file.csv" --output-file="my_directory/form8949_2025.csv"
```

### Option B: Docker
Docker users: see [Quickstart (alternative)](#quickstart-alternative-docker): Docker.

## Compare your output to the expected tutorial results

Tutorial mode is designed to be deterministic so you can validate behavior by comparing your results
against committed “golden” results.

### Run tutorial manually
A tutorial case is available to ensure that the Form 8949 is being created correctly.  To create the 
corresponding form, the following should be run from the repo root:
```bash
irs-fifo-taxes --input-file=examples/asset_tx.csv --output-file=form8949_example.csv
bash scripts/compare-tutorial-results.sh
```
If there are no executable permissions on the script, they can be set with
```bash
chmod u+x scripts/compare-tutorial-results.sh
```

### Compare using the provided script
From the repository root:
```bash
bash scripts/compare-tutorial-logs.sh
```

### Expected results (golden files)

The expected tutorial logs are stored in the repository at:
- `examples/asset_tx.csv`
- `examples/form8949.csv`

## Technologies

IRS asset FIFO calculator uses the following technologies and tools:

- [![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org/)
- [![Sphinx](https://img.shields.io/badge/Sphinx-3B4D92?style=for-the-badge&logo=sphinx&logoColor=white)](https://www.sphinx-doc.org/en/master/)

## Development
### Building the docs in PyCharm

In order to create Sphinx documentation from the docstrings in PyCharm, a new run task must be created: 
Run > Edit Configurations... > + (top-left) > Sphinx task.  In the window that opens, name the Sphinx task in the
"Name" field, select "html" under the "Command:" dropdown, select the docs folder in the root folder in the "Input:"
field, and select the docs/_build folder in the "Output:" field.  If the docs or docs/_build folder do not already
exist, they will perhaps need to be created.  The Sphinx documentation can now be created by going to Run > Run... and
selecting the Sphinx task name.

## Contributing

To contribute to the development of IRS asset FIFO calculator, follow the steps below:

1. Fork IRS asset FIFO calculator from <https://github.com/elliottbache/irs_asset_fifo_calculator/fork>
2. Create your feature branch (`git checkout -b feature-new`)
3. Make your changes
4. Commit your changes (`git commit -am 'Add some new feature'`)
5. Push to the branch (`git push origin feature-new`)
6. Create a new pull request

## Contributors

Here's the list of people who have contributed to IRS asset FIFO calculator:

- Elliott Bache – elliottbache@gmail.com

The IRS asset FIFO calculator development team really appreciates and thanks the time and effort that all
these fellows have put into the project's growth and improvement.

## Author

- Elliott Bache – elliottbache@gmail.com

## Change log

- 0.1.0
    - First public FIFO release

## License

IRS asset FIFO calculator is distributed under the GPL-3.0 license. 

<!-- docs:end -->
