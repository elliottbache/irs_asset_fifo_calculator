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

## Installation
No installation is required.  

## Quick start

1. Put your transactions in `asset_tx.csv` with the header:
   `Date, Tx Index, Asset, Amount (asset), Sell price ($), Buy price ($), Type, ...`
2. Run from CLI:
```bash
cd src
python irs-fifo-taxes
```

### Tutorial case
A tutorial case is available to ensure that the Form 8949 is being created correctly.  To create the 
corresponding form, the following should be run from the repo root:
```bash
python irs-fifo-taxes --input-file=examples/asset_tx.csv --output-file=form8949_example.csv
bash scripts/compare-tutorial-results.sh
```
If there are no executable permissions on the script, they can be set with
```bash
chmod u+x scripts/compare-tutorial-results.sh
```

## Technologies

IRS asset FIFO calculator uses the following technologies and tools:

- [![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org/)
- [![Sphinx](https://img.shields.io/badge/Sphinx-3B4D92?style=for-the-badge&logo=sphinx&logoColor=white)](https://www.sphinx-doc.org/en/master/)

## Development
### Building the docs

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
