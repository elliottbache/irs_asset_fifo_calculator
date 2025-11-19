"""
Calculate IRS capital gains taxes using FIFO method.

Tax calculator that tracks capital gains from multiple purchases and sales.
This program uses a CSV file as input.  This file is called "asset_tx.csv"
in the published example, but any name can be used, using this name in
the python call. The file has the following header: Date,Asset,Amount (asset),
Sell Price ($),Buy Price ($),Account number,Entity,Notes,Remaining

Functions:
    record_sale:
        Write a sale to the Form 8949 file object.

    main:
        Main function.

"""

def record_sale(asset, amount, proceeds, cost_basis, acquisition_date, sale_date):
    """Record a sale.

    This takes various data about the sale appends the data to the open Form 8949 file object.

    Args:
        asset (str): The asset name.
        amount (str): The amount of the asset units.
        proceeds (int): The dollar amount of capital gains.
        cost_basis (str): The dollar cost of this amount of asset.
        acquisition_date (str): The acquisition date.
        sale_date (str): The sale date.

    Returns:
        int : 0 if no error.
    """

    if proceeds >= 0.005 or cost_basis >= 0.005:
        form8949.append({
            "Description": f"{amount:.8f} {token}",
            "Date Acquired": acquisition_date.strftime("%m/%d/%Y"),
            "Date Sold": sale_date.strftime("%m/%d/%Y"),
            "Proceeds": f"{proceeds:.2f}",
            "Cost Basis": f"{cost_basis:.2f}",
            "Gain or Loss": f"{proceeds - cost_basis:.2f}",
            "Code": "",
            "Adjustment Amount": ""
        })
    return 0