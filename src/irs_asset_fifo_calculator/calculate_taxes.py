"""
Calculate IRS capital gains taxes using FIFO method.

Tax calculator that tracks capital gains from multiple purchases and
sales.  This program uses a CSV file as input.  This file is called
"asset_tx.csv" in the published example, but any name can be used,
using this name in the python call. The file has the following header:
Date,Asset,Amount (asset), Sell Price ($),Buy Price ($),Account number,
Entity,Notes,Remaining

Functions:
    record_sale:
        Write a sale to the Form 8949 file object.

    main:
        Main function.

"""

import sys
from datetime import datetime
from typing import List, Dict, DefaultDict, Union, Deque
from collections import defaultdict, deque


def record_sale(form8949: List[Dict[str, str]], asset: str, amount: float,
                proceeds: float, cost_basis: float, acquisition_date: datetime,
                sale_date: datetime) -> None:
    """Record a sale.

    This takes various data about the sale appends the data to the
    open Form 8949 file object.

    Args:
        form8949 (List[Dict[str, str]]): Form 8949 list of dicts
         holding txs.
        asset (str): The asset name.
        amount (float): The amount of the asset units.
        proceeds (float): The dollar amount of capital gains.
        cost_basis (float): The dollar cost of this amount of asset
            taking into account purchase fees.
        acquisition_date (datetime): The acquisition date.
        sale_date (datetime): The sale date.

    Returns:
        None.

    Example:
        >>> from calculate_taxes import record_sale
        >>> from datetime import datetime
        >>> form8949 = list()
        >>> form8949.append({"Description": "10.00000000 NVDA",
        ...     "Date Acquired": "1982-10-27",
        ...     "Date Sold": "2024-12-31",
        ...     "Proceeds": "10000",
        ...     "Cost Basis": "1000",
        ...     "Gain or Loss": "9000",
        ...     "Code": "",
        ...     "Adjustment Amount": ""})
        >>> record_sale(form8949, "TSLA", 10, 100, 90, datetime(2024,1,1),
        ...     datetime(2024,12,31))
        >>> len(form8949)
        2
        >>> form8949[1]["Description"]
        '10.00000000 TSLA'
        >>> form8949[1]["Date Acquired"]
        '2024-01-01'
        >>> form8949[1]["Date Sold"]
        '2024-12-31'
        >>> form8949[1]["Proceeds"]
        '100.00'
        >>> form8949[1]["Cost Basis"]
        '90.00'
        >>> form8949[1]["Gain or Loss"]
        '10.00'
        >>> form8949[1]["Code"]
        ''
        >>> form8949[1]["Adjustment Amount"]
        ''
    """

    if not isinstance(acquisition_date, datetime):
        raise TypeError(
            "Acquisition date must be in datetime format.\n"
            + str(amount) + " " + asset + " purchase on "
            + acquisition_date + " is invalid."
        )
    if not isinstance(sale_date, datetime):
        raise TypeError(
            "Sale date must be in datetime format.\n",
            amount, " ", asset, "sale on ", sale_date, " is invalid."
        )

    if not ((isinstance(amount, float) or isinstance(amount, int)) and
            (isinstance(proceeds, float) or isinstance(proceeds, int)) and
            (isinstance(cost_basis, float) or isinstance(cost_basis, int))):
        print(type(amount), type(proceeds), type(cost_basis))
        raise TypeError(
            "Amounts ($ and asset) must be in float format.\n"
            + str(amount) + " " + asset + " sale on "
            + sale_date.strftime("%Y-%m-%d") + " is invalid."
        )

    if amount < 0:
        print("Amount must be greater than zero.\n",
              amount, " ", asset, "sale on ", sale_date, "is set as absolute.")
        amount = abs(amount)
    if proceeds < 0:
        print("Proceeds must be greater than zero.\n",
              amount, " ", asset, "sale on ", sale_date, "is set as absolute.")
        proceeds = abs(proceeds)
    if cost_basis < 0:
        print("Cost basis must be greater than zero.\n",
              amount, " ", asset, "purchase on ", acquisition_date,
              "is set as absolute.")
        cost_basis = abs(cost_basis)

    if not isinstance(form8949, list):
        raise TypeError(
            "A list object must be passed. Create form8949 list first."
        )

    if acquisition_date > sale_date:
        raise ValueError(
            "Acquisition date must be before sale date.\n"
            + str(amount) + " " + asset + " sale on " + str(sale_date)
            + " is invalid."
        )

    if proceeds >= 0.005 or cost_basis >= 0.005:

        form8949.append({
            "Description": f"{round(amount,8):.8f}" + " " + asset,
            "Date Acquired": acquisition_date.strftime("%Y-%m-%d"),
            "Date Sold": sale_date.strftime("%Y-%m-%d"),
            "Proceeds": f"{round(proceeds,2):.2f}",
            "Cost Basis": f"{round(cost_basis,2):.2f}",
            "Gain or Loss": f"{round(proceeds - cost_basis,2):.2f}",
            "Code": "",
            "Adjustment Amount": ""
        })


def update_fifo(
        form8949: List[Dict[str, str]], sell_amount: float, asset: str,
        fifo: DefaultDict[str, Deque[Dict[str, Union[float | datetime]]]],
        proceeds: float,
        timestamp: datetime) -> None:
    """ Update fifo list of lists.

    Takes a sale and reduces the FIFO dict by the sale amount.

    Args:
        form8949 (List[Dict[str, str]]): Form 8949 list of dicts
            holding txs.
        sell_amount (float): this sale's amount
        asset (str): this asset
        fifo (DefaultDict[str, Deque[Dict[str, Union[float | datetime]]]]):
            purchases of each token defined by their amount, price,
            cost, and date
        proceeds (float): this sale's proceeds
        timestamp (datetime): this sale's date

    Returns:
        None

    Example:
        >>> from calculate_taxes import update_fifo
        >>> from datetime import datetime
        >>> form8949 = list()
        >>> fifo = defaultdict(deque)
        >>> fifo['NVDA'].append({"amount": 10, "price": 10,
        ...     "cost": 100*1.002, "timestamp": datetime(2024, 1, 1)})
        >>> fifo['NVDA'].append({"amount": 20, "price": 11,
        ...     "cost": 210*1.002, "timestamp": datetime(2024, 2, 1)})
        >>> update_fifo(form8949, 15, 'NVDA', fifo, 135,
        ...     datetime(2024, 3, 1))
        >>> len(fifo['NVDA'])
        1
        >>> abs(fifo['NVDA'][0]['amount'] - 15) < 0.001
        True
        >>> abs(fifo['NVDA'][0]['price'] - 11) < 0.001
        True
        >>> abs(fifo['NVDA'][0]['cost'] - 157.5*1.002) < 0.001
        True
        >>> fifo['NVDA'][0]['timestamp']
        datetime.datetime(2024, 2, 1, 0, 0)
    """

    remaining = sell_amount
    while remaining > 0 and fifo[asset]:

        # set the current lot
        lot = fifo[asset][0]
        lot_amount = lot['amount']

        used = min(remaining, lot_amount)
        acquisition_date = lot['timestamp']

        # proportional cost and proceeds from used
        this_cost = used / lot_amount * lot['cost']
        this_proceeds = used / sell_amount * proceeds

        record_sale(form8949, asset, used, this_proceeds, this_cost,
                    acquisition_date, timestamp)

        lot['amount'] -= used
        lot['cost'] -= this_cost
        if lot['amount'] == 0:
            fifo[asset].popleft()

        remaining -= used


if __name__ == "__main__":
    form8949 = []
    asset = 'AMZN'
    amount = 10
    proceeds = 100
    price = 10
    cost_basis = 100
    acquisition_date = datetime(2024, 1, 1)
    sale_date = datetime.now()
    """
    acquisition_date = datetime.now()
    sale_date = datetime(2024, 1, 1)
    """

    try:
        record_sale(form8949, asset, amount, proceeds, cost_basis,
                    acquisition_date, sale_date)
    except (ValueError, TypeError) as err:
        print(f"Error recording sale: {err}")
        sys.exit(1)

    print(form8949)

    fifo = defaultdict(deque)
    fifo[asset].append({
        "amount": amount,
        "price": price,
        "cost": cost_basis*1.002,
        "timestamp": acquisition_date
    })
    fifo[asset].append({
        "amount": amount*0.5,
        "price": price*1.1,
        "cost": (amount*0.5 * price*1.1)*1.002,
        "timestamp": datetime(2024,2,1)
    })
    fifo[asset].append({
        "amount": amount*0.2,
        "price": price*0.8,
        "cost": (amount*0.2 * price*0.8)*1.002,
        "timestamp": datetime(2024,3,1)
    })
    print(fifo)

    update_fifo(form8949, 5, asset, fifo, 60, datetime(2024,4,1))
    print(fifo)

    update_fifo(form8949, 6, asset, fifo, 70, datetime(2024,4,2))
    print(fifo)
