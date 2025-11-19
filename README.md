<!-- docs:start -->
# IRS asset FIFO calculator

# To do!!!
- Replace readthedocs link
- Check if all columns in CSV are needed and change description accordingly.  Also change where the columns
are treated inside the code.
- Add arguments to python call.

<!---
[![Documentation Status](https://readthedocs.org/projects/tls-connection-coding-challenge/badge/?version=latest)](https://tls-connection-coding-challenge.readthedocs.io/en/latest/?badge=latest)
-->

Tax calculator that tracks capital gains from multiple purchases and sales.  This program uses a CSV file as input.  

This file is called "asset_tx.csv" in the published example, but any name can be
be used, using this name in the python call.  The file has the following header:
Date,Asset,Amount (asset),Sell Price ($),Buy price ($),Account number,Entity,Notes,Remaining

**Table of Contents**

- [Installation](#installation)
- [Execution / Usage](#execution--usage)
- [Technologies](#technologies)
- [Contributing](#contributing)
- [Contributors](#contributors)
- [Author](#author)
- [Change log](#change-log)
- [License](#license)

## Installation
No installation is required.  

## Execution / Usage
This program was developed with Python 3.11.14.  To run this tax calculator, only the python file and the input CSV file
containing all of the transactions are needed.
```sh
$ python main.py asset_tx.csv
```

## Technologies

IRS asset FIFO calculator uses the following technologies and tools:

- [Python](https://www.python.org/): ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

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

- 0.0.1
    - First working version

## License

IRS asset FIFO calculator is distributed under the MIT license. 

<!-- docs:end -->