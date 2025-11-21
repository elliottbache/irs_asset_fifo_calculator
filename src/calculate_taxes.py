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
from typing import List, Dict

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
        cost_basis (float): The dollar cost of this amount of asset.
        acquisition_date (datetime): The acquisition date.
        sale_date (datetime): The sale date.

    Returns:
        None.
    """

    if not isinstance(acquisition_date, datetime):
        raise TypeError("Acquisition date must be in datetime format.\n"
                + str(amount) + " " + asset + " purchase on " + acquisition_date
                + " is invalid.")
    if not isinstance(sale_date, datetime):
        raise TypeError("Sale date must be in datetime format.\n",
                amount, " ", asset, "sale on ", sale_date, " is invalid.")

    if not ((isinstance(amount, float) or isinstance(amount, int)) and
            (isinstance(proceeds, float) or isinstance(proceeds, int)) and
            (isinstance(cost_basis, float) or isinstance(cost_basis, int))):
        print(type(amount), type(proceeds), type(cost_basis))
        raise TypeError("Amounts ($ and asset) must be in float format.\n"
              + str(amount) + " " + asset + " sale on " + sale_date.strftime("%Y-%m-%d") + " is invalid.")

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
              amount, " ", asset, "purchase on ", acquisition_date, "is set as absolute.")
        cost_basis = abs(cost_basis)

    if type(form8949) != list:
        raise TypeError("A list object must be passed.  Create form8949 list first.")

    if acquisition_date > sale_date:
        raise ValueError("Acquisition date must be before sale date.\n"+
            str(amount) + " " + asset + " sale on " + str(sale_date) + " is invalid.")

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

if __name__ == "__main__":
    form8949 = []
    asset = 'AMZN'
    amount = 10
    proceeds = 100
    cost_basis = 90
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
