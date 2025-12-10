"""
Calculate IRS capital gains taxes using FIFO method.

Tax calculator that tracks capital gains from multiple purchases and
sales.  This program uses a CSV file as input.  This file is called
"asset_tx.csv" in the published example, but any name can be used,
using this name in the python call. The file has the following header:
Tx Index, Date, Asset, Amount (asset), Sell price ($), Buy price ($),
Type, Account number, Entity, Notes, Remaining

Transfer fees are not deducted although if paid with an asset,
capital gains apply on the conversion of the asset to USD.

Functions:
    record_sale:
        Write a sale to the Form 8949 file object.
    is_finite_number:
        Return True if x is a finite (non-NaN, non-infinite) real
        number.
    reduce_fifo:
        Update FIFO lots for a sale.
    parse_amount:
        Parse amount from input.  Can be string or numeric.
    is_fee:
        Check if asset is a fee transaction.
    parse_buy_and_sell:
        Extract the buy or sell side from a block of related
        transactions.
    parse_row_data:
        Extract the necessary values from row data.
    update_fifo:
        Updates FIFO dict of deques using info from this block of
        transactions.

    main:
        Main function.

"""

from datetime import date
from math import isfinite, isclose
from typing import List, Dict, DefaultDict, Deque, TypedDict, Any, Literal
from collections import defaultdict, deque
import numbers
import pandas as pd
from dataclasses import dataclass

def record_sale(form8949: List[Dict[str, str]], asset: str, amount: float,
                proceeds: float, cost_basis: float, acquisition_date: date,
                sale_date: date) -> None:
    """Record a sale.

    This takes various data about the sale and appends the data to the
    open Form 8949 file object.

    Args:
        form8949 (List[Dict[str, str]]): Form 8949 list of dicts
         holding txs.
        asset (str): The asset name.
        amount (float): The amount of the asset units.
        proceeds (float): The dollar amount of capital gains.
        cost_basis (float): The dollar cost of this amount of asset
            taking into account purchase fees.
        acquisition_date (date): The acquisition date.
        sale_date (date): The sale date.

    Returns:
        None.

    Example:
        >>> from calculate_taxes import record_sale
        >>> from datetime import date
        >>> form8949 = list()
        >>> form8949.append({"Description": "10.00000000 NVDA",
        ...     "Date Acquired": "11/28/1982",
        ...     "Date Sold": "12/31/2024",
        ...     "Proceeds": "10000",
        ...     "Cost Basis": "1000",
        ...     "Gain or Loss": "9000"})
        >>> record_sale(form8949, "TSLA", 10, 100, 90, date(2024,1,1),
        ...     date(2024,12,31))
        >>> len(form8949)
        2
        >>> form8949[1]["Description"]
        '10.00000000 TSLA'
        >>> form8949[1]["Date Acquired"]
        '01/01/2024'
        >>> form8949[1]["Date Sold"]
        '12/31/2024'
        >>> form8949[1]["Proceeds"]
        '100.00'
        >>> form8949[1]["Cost Basis"]
        '90.00'
        >>> form8949[1]["Gain or Loss"]
        '10.00'
    """

    if not isinstance(acquisition_date, date):
        raise TypeError(
            f"Acquisition date must be in date format.\n"
            f"{str(amount)} {asset} purchase on {acquisition_date} is invalid."
        )
    if not isinstance(sale_date, date):
        raise TypeError(
            f"Sale date must be in date format.\n"
            f"{amount} {asset} sale on {sale_date} is invalid."
        )

    for name, value in (("amount", amount), ("proceeds", proceeds),
                        ("cost_basis", cost_basis)):
        if not is_finite_number(value):
            raise TypeError(f"{name} is not a valid number: {value}."
                            f" sale_date: {sale_date} asset: {asset} amount: {amount}")

    if amount < 0:
        print("Amount must be greater than zero.\n",
              amount, " ", asset, "sale on ", sale_date, "is set as absolute.")
        amount = abs(amount)

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

        # place negative numbers in parentheses
        if proceeds - cost_basis < 0:
            gain_or_loss = f"({round(abs(proceeds - cost_basis),2):.2f})"
        else:
            gain_or_loss = f"{round(proceeds - cost_basis,2):.2f}"

        form8949.append({
            "Description": f"{round(amount,8):.8f}" + " " + asset,
            "Date Acquired": acquisition_date.strftime("%m/%d/%Y"),
            "Date Sold": sale_date.strftime("%m/%d/%Y"),
            "Proceeds": f"{round(proceeds,2):.2f}",
            "Cost Basis": f"{round(cost_basis,2):.2f}",
            "Gain or Loss": gain_or_loss
        })

def is_finite_number(x: object) -> bool:
    """Return True if x is a finite (non-NaN, non-infinite) real number."""
    # 1) Is it a numeric type? (int, float, Decimal, Fraction, etc.)
    if not isinstance(x, numbers.Number):
        return False

    # Optional: exclude bool, since bool is a subclass of int
    if isinstance(x, bool):
        return False

    # 2) Is it finite (not NaN, +inf, -inf)?
    return isfinite(float(x))

class FifoLot(TypedDict):
    amount: float
    price: float
    cost: float
    tx_date: date

def reduce_fifo(
        form8949: List[Dict[str, str]], sell_amount: float, asset: str,
        fifo_asset: Deque[FifoLot],
        proceeds: float,
        sale_date: date) -> None:
    """Update FIFO lots for a sale.

    Takes a sale and reduces the FIFO lots for that asset by the sale
    amount, recording one or more rows in form8949.

    Args:
        form8949 (List[Dict[str, str]]): Form 8949 list of dicts
            holding txs.
        sell_amount (float): this sale's amount
        asset (str): this asset
        fifo_asset (Deque[FifoLot]):
            purchases for this token defined by their amount, price,
            cost, and date
        proceeds (float): this sale's proceeds
        sale_date (date): this sale's date

    Returns:
        None

    Example:
        >>> from calculate_taxes import reduce_fifo
        >>> from datetime import date
        >>> from collections import defaultdict, deque
        >>> form8949 = list()
        >>> fifo = defaultdict(deque)
        >>> fifo['NVDA'].append({"amount": 10, "price": 10,
        ...     "cost": 100*1.002, "tx_date": date(2024, 1, 1)})
        >>> fifo['NVDA'].append({"amount": 20, "price": 11,
        ...     "cost": 210*1.002, "tx_date": date(2024, 2, 1)})
        >>> reduce_fifo(form8949, 15, 'NVDA', fifo['NVDA'], 135,
        ...     date(2024, 3, 1))
        >>> len(fifo['NVDA'])
        1
        >>> abs(fifo['NVDA'][0]['amount'] - 15) < 0.001
        True
        >>> abs(fifo['NVDA'][0]['price'] - 11) < 0.001
        True
        >>> abs(fifo['NVDA'][0]['cost'] - 157.5*1.002) < 0.001
        True
        >>> fifo['NVDA'][0]['tx_date']
        datetime.date(2024, 2, 1)
    """

    if sell_amount < 0:
        sell_amount = abs(sell_amount)

    amount_tol = 5e-9 # tolerance for remaining asset amount
    proceeds_tol = 5e-3 # tolerance for remaining asset proceeds
    remaining = sell_amount
    while remaining > amount_tol and fifo_asset:

        # set the current lot
        lot = fifo_asset[0]

        # check if all necessary keys are present in fifo row
        required_keys = ['amount', 'price', 'cost', 'tx_date']
        if not all(key in lot for key in required_keys):
            raise KeyError(f"FIFO contains an invalid purchase. {lot}")

        if not isinstance(lot['tx_date'], date):
            raise TypeError(f"FIFO contains an invalid purchase date: {lot}.")

        for name, value in (("amount", lot['amount']), ("price", lot['price']),
                            ("cost", lot['cost'])):
            if not is_finite_number(value):
                raise TypeError(f"{name} is not a valid number: {value}.")

        if lot['amount'] == 0:
            fifo_asset.popleft()
            continue
        elif lot['amount'] < 0:
            raise ValueError(f"FIFO amount is negative for sale: {lot}.")

        if lot['cost'] < 0:
            raise ValueError(f"FIFO cost is negative for sale: {lot}.")

        acquisition_date = lot['tx_date']
        used = min(remaining, lot['amount'])

        # proportional cost and proceeds from used
        this_cost = used / lot['amount'] * lot['cost']

        this_proceeds = used / sell_amount * proceeds

        record_sale(form8949, asset, used, this_proceeds, this_cost,
                    acquisition_date, sale_date)

        lot['amount'] -= used
        if lot['amount'] == 0:
            fifo_asset.popleft()

        lot['cost'] -= this_cost

        remaining -= used

    # make sure remaining amount in $ is less than 0.01
    if remaining * max(1e-8, proceeds / sell_amount) > proceeds_tol:
        raise ValueError(
            f"Not enough {asset} to sell: remaining {remaining} after "
            f"exhausting FIFO lots."
        )

def parse_amount(value: Any) -> float:
    """Parse amount from input.  Can be string or numeric.
    Extra whitespace is valid, $ or € signs are not."""

    if isinstance(value, numbers.Number):
        return float(value)

    if isinstance(value, str):
        clean_value = "".join(value.replace(',', '').split())
        try:
            return float(clean_value)
        except ValueError:
            raise ValueError(f"Invalid amount {value}")

    raise TypeError(f"Invalid amount {value}")

def is_fee(asset: str | None) -> bool:
    """Check if asset is a fee transaction.

    In order to be a fee, the asset must start with the letters 'fee',
    and be longer than 3 characters.
    """
    if asset.startswith("fee") and len(asset) > 3:
        return True
    else:
        return False


@dataclass
class AssetData:
    asset: str | None
    amount: float
    price: float
    total: float
    tx_date: date


BlockType = Literal['Buy', 'Sell', 'Exchange', 'Transfer']

def parse_buy_and_sell(is_buy: bool, block_type: str, rows: pd.DataFrame, fee_assets: set[str], fee_rows: List[int]) -> tuple[str | None, float, float, float]:
    """Extract the buy or sell side from a block of related transactions.

    For non-transfer blocks, this scans the rows (excluding fee rows) to
    find the one representing the buy or sell side based on the sign of
    'Amount (asset)' (is_buy = True: amount > 0, is_buy = False: amount < 0).
    It returns that row's asset symbol, signed amount, and unit price.
    If the resulting asset is in fee_assets, any fee rows are added
    to the returned amount.
    - The price for USD is always forced to 1.0.
    - Returns (None, 0.0, 0.0, 0.0) if
        - block_type == 'Transfer'
        - no matching non-fee row is found

    Args:
        is_buy (bool): are we parsing buy side?
        block_type (str): Type of block ('Exchange', 'Buy', 'Sell',
            or 'Transfer')
        rows (pd.DataFrame): the transactions for this block. Must
            include at least 'Asset', 'Amount (asset)', 'Buy price ($)',
            and 'Sell price ($)'
        fee_assets (set[str]): assets that have associated fee rows
            (e.g. 'USD', not 'feeUSD')
        fee_rows (list[int]): indices within rows that correspond to fee
            transactions

    Returns:
        (tuple[str, float, float, float]): asset, amount, price, cost or proceeds

    Raises:
        ValueError: If more than one non-fee row matches the requested side.
        ValueError: If the fee asset is the same as the buy or sell asset but the
            prices are different.

    Example:
        >>> import pandas as pd
        >>> from datetime import date
        >>> from calculate_taxes import parse_buy_and_sell
        >>> rows = pd.DataFrame([
        ...     {"Tx Date": date(2024, 9, 4), "Asset": "TSLA",
        ...      "Amount (asset)": -25.0, "Sell price ($)": 50.0,
        ...      "Buy price ($)": float("nan"), "Type": "Exchange"},
        ...     {"Tx Date": date(2024, 9, 4), "Asset": "NVDA",
        ...      "Amount (asset)": 10.0, "Sell price ($)": float("nan"),
        ...      "Buy price ($)": 125.0, "Type": "Exchange"},
        ...     {"Tx Date": date(2024, 9, 4), "Asset": "feeUSD",
        ...      "Amount (asset)": -10.0, "Sell price ($)": 1.0,
        ...      "Buy price ($)": float("nan"), "Type": "Exchange"},
        ... ])
        >>> fee_rows = [2]
        >>> fee_assets = ["USD"]
        >>> parse_buy_and_sell(True, "Exchange", rows, fee_assets, fee_rows)
        ('NVDA', 10.0, 125.0, 1260.0)
        >>> parse_buy_and_sell(False, "Exchange", rows, fee_assets, fee_rows)
        ('TSLA', -25.0, 50.0, 1240.0)
    """

    if block_type == 'Transfer':
        return None, 0.0, 0.0, 0.0

    # identify row with buy or sell data
    buy_or_sell_idx = None
    for idx in range(len(rows)):
        amount = parse_amount(rows.iloc[idx]['Amount (asset)'])
        if idx not in fee_rows and \
                ((not is_buy and amount < 0) or (is_buy and amount > 0)):
            if buy_or_sell_idx is not None:
                raise ValueError(f"Multiple rows for buy or sell must be implemented {rows}.")
            buy_or_sell_idx = idx

    # define buy or sell data
    if is_buy:
        which_price =  'Buy price ($)'
    else:
        which_price = 'Sell price ($)'

    if buy_or_sell_idx is None:
        buy_or_sell_asset = None
        buy_or_sell_amount = 0.0
        buy_or_sell_price = 0.0
    else:
        row = rows.iloc[buy_or_sell_idx]
        buy_or_sell_asset = row['Asset']
        buy_or_sell_amount = parse_amount(row['Amount (asset)']) # negative
        buy_or_sell_price = parse_amount(row[which_price])

    # Dollar is always worth 1 dollar
    if buy_or_sell_asset == 'USD':
        buy_or_sell_price = 1.0

    # calculate cost or proceeds before adding fees
    cost_or_proceeds = abs(buy_or_sell_amount * buy_or_sell_price)

    # Add all fees in the buy or sell asset to the buy or sell amount
    if buy_or_sell_asset in fee_assets:
        for idx in range(len(fee_rows)):
            row = rows.iloc[fee_rows[idx]]
            if row['Asset'] == 'fee' + buy_or_sell_asset:
                buy_or_sell_amount += parse_amount(row['Amount (asset)'])

    # add all fees to cost and proceeds (not only fee assets that are
    # same as buy and sell)
    for idx in range(len(fee_rows)):
        row = rows.iloc[fee_rows[idx]]

        # make sure fee price is same as buy or sell price if the same asset
        if buy_or_sell_asset is not None and row['Asset'] == 'fee' + buy_or_sell_asset:
            if not isclose(buy_or_sell_price, parse_amount(row['Sell price ($)']), rel_tol=1e-6):
                raise ValueError(f"Fee price does not match buy or sell price for \n{row} \n\nin \n{rows}.")

        if is_buy:
            cost_or_proceeds += abs(parse_amount(row['Amount (asset)'])) * parse_amount(row['Sell price ($)'])
        else:
            cost_or_proceeds -= abs(parse_amount(row['Amount (asset)'])) * parse_amount(row['Sell price ($)'])

    return buy_or_sell_asset, buy_or_sell_amount, buy_or_sell_price, cost_or_proceeds

def parse_row_data(block_type: BlockType, rows: pd.DataFrame) -> tuple[AssetData, AssetData, AssetData]:
    """Extract the necessary values from row data.

    Notes:
    - Transfer fees are not deducted although if paid with an asset, the
        conversion of the asset to USD is taxed.
    - For transfers, there is no buy data.  The fee data becomes the
        sell data.
    - Proceeds from fee assets are:
        - Added to cost if the fee asset is the same as the bought asset
        - Deducted from proceeds if the fee asset is the same as the sold asset
        - Recorded as a sale if it is a transfer or if they are different from
            both assets in a buy/sell/exchange.
    - Here we assume that there con only be a maximum of 1 fee asset
        besides the buy and sell assets. Sell and fee amount are
        negative in general.
    - If the fee asset is the same as the buy or sell asset, it is
        included in these, and the fee amount for that asset
        is set to 0.  If there are no other fee assets, then the fee
        asset will be None.
    - With large enough fees, the buy amount may become negative, in
        which case it will later be used to update FIFO (reduce and
        append to form8949) rather than append to FIFO.

    Args:
        block_type (BlockType): The type of block to extract from.  Can
            take the following values: ['Buy', 'Sell',
            'Exchange', 'Transfer']
        rows (pd.DataFrame): The row data to extract from. The mandatory
            columns are: [Tx Index, Tx Date, Asset, Amount (asset), Buy price ($),
            Sell price ($), Type]

    Returns:
        AssetData, AssetData, AssetData: buy data, sell data, and fee
            data.


    Example:
        >>> import pandas as pd
        >>> from datetime import date
        >>> from calculate_taxes import parse_row_data
        >>> block_type = 'Buy'
        >>> rows = pd.DataFrame({'Tx Index': [0] * 3, 'Tx Date': [date(2024, 9, 4)] * 3,
        ...     'Asset': ['USD', 'NVDA', 'feeUSD'],
        ...     'Amount (asset)': [-1250, 10, -10],
        ...     'Sell price ($)': [1, 'NaN', 1],
        ...     'Buy price ($)': [1, 125, 1],
        ...     'Type': ['Buy'] * 3})
        >>> buy_data, sell_data, fee_data = parse_row_data(block_type, rows)
        >>> buy_data
        AssetData(asset='NVDA', amount=10.0, price=125.0, total=1260.0, tx_date=datetime.date(2024, 9, 4))
        >>> sell_data
        AssetData(asset='USD', amount=-1260.0, price=1.0, total=0.0, tx_date=datetime.date(2024, 9, 4))
        >>> fee_data
        AssetData(asset='USD', amount=-10.0, price=1.0, total=10.0, tx_date=datetime.date(2024, 9, 4))
    """

    if block_type not in ['Buy', 'Sell', 'Exchange', 'Transfer']:
        raise ValueError(f"{block_type} is not a valid block type")

    # change to date format
    raw_date = rows.iloc[0]['Tx Date']
    if hasattr(raw_date, 'date'):
        first_date = raw_date.date()
    else:
        first_date = raw_date

    # identify fee rows and assets
    fee_rows = []
    fee_assets = set()
    for idx in range(len(rows)):
        if is_fee(rows.iloc[idx]['Asset']):
            fee_rows.append(idx)
            fee_assets.add(rows.iloc[idx]['Asset'][len('fee'):])

    buy_asset, buy_amount, buy_price, cost = parse_buy_and_sell(True, block_type, rows, fee_assets, fee_rows)
    sell_asset, sell_amount, sell_price, proceeds = parse_buy_and_sell(False, block_type, rows, fee_assets, fee_rows)
    if buy_asset is not None and buy_asset == sell_asset:
        raise ValueError("Buy and sell asset cannot be the same.")

    # remove buy and sell assets from fee assets
    if buy_asset in fee_assets: fee_assets.remove(buy_asset)
    if sell_asset in fee_assets: fee_assets.remove(sell_asset)
    idx = 0
    while idx < len(fee_rows):
        if rows.iloc[fee_rows[idx]]['Asset'][len('fee'):] in [buy_asset, sell_asset]:
            del fee_rows[idx]
        else:
            idx += 1

    # check that there is max of 1 fee asset different from
    # buy and sell assets
    if len(fee_assets) > 1:
        raise ValueError(f"Too many fee assets: {fee_assets} in {rows}.")

    fee_asset = None
    fee_amount, fee_price = 0.0, 0.0
    if len(fee_assets) == 1:
        fee_asset = next(iter(fee_assets))
        # fee_price is an average even though all fee_price for the
        # same tx should be the same
        for idx in range(len(fee_rows)):
            this_amount = parse_amount(rows.iloc[fee_rows[idx]]['Amount (asset)'])
            fee_amount += this_amount
            fee_price += this_amount * parse_amount(rows.iloc[fee_rows[idx]]['Sell price ($)'])
        if fee_amount == 0:
            fee_price = 0.0
        else:
            fee_price /= fee_amount

    fee_proceeds = -fee_amount * fee_price # positive

    # define cost with using USD amounts when available, otherwise use buy data
    # proceeds are 0 for transfers and purchases (USD doesn't give gains)
    if cost < 0:
        raise ValueError(f"Cost cannot be negative: {cost} for {rows}")

    buy_data = AssetData(asset=buy_asset, amount = float(buy_amount),
                         price=float(buy_price), total=float(cost),
                         tx_date=first_date)
    sell_data = AssetData(asset=sell_asset, amount=float(sell_amount),
                          price=float(sell_price), total=float(proceeds),
                          tx_date=first_date)
    fee_data = AssetData(asset=fee_asset, amount=float(fee_amount),
                         price=float(fee_price), total=float(fee_proceeds),
                         tx_date=first_date)

    return buy_data, sell_data, fee_data

def update_fifo(buy_data: AssetData, sell_data: AssetData, fee_data: AssetData,
                form8949: List[Dict[str, str]],
                fifo: DefaultDict[str, Deque[FifoLot]]) -> None:
    """Updates FIFO dict of deques using info from this block of transactions.

    Notes:
    - In general, buy and sell assets and fee asset should not be the
    same.  If they were upstream, the fees were added to buy or sell
    and then set to 0.
    - If previously calculated fees are same asset as buy and larger than
    buy amount, the net buy amount is negative and is thus reduced from
    FIFO instead of appended.

    Args:
        buy_data (AssetData): buy info for this block of transactions
        sell_data (AssetData): sell info for this block of transactions
        fee_data (AssetData): fee info for this block of transactions
        form8949 (List[Dict[str, str]]): Form 8949 list of dicts
         holding txs.
        fifo (DefaultDict[str, Deque[FifoLot]]):
            purchases of each token defined by their amount, price,
            cost, and date


    Returns:
        None

    Example:

    """

    if buy_data.asset is not None and buy_data.asset != 'USD':
        if buy_data.amount > 0:
            fifo[buy_data.asset].append(
                {"amount": buy_data.amount, "price": buy_data.price,
                 "cost": buy_data.total, "tx_date": buy_data.tx_date}
            )
        elif buy_data.amount < 0: # if fees exceed buy amount
            reduce_fifo(form8949, abs(buy_data.amount), buy_data.asset,
                        fifo[buy_data.asset], buy_data.total, buy_data.tx_date)

    if (sell_data.asset is not None and sell_data.asset != 'USD' and
            sell_data.amount < 0):
        reduce_fifo(form8949, abs(sell_data.amount), sell_data.asset,
                    fifo[sell_data.asset], sell_data.total, sell_data.tx_date)

    # if they are the same, the fees are already taken into account in
    # the buy_data and sell_data
    if fee_data.asset == buy_data.asset or fee_data.asset == sell_data.asset:
        raise ValueError(f"Fee asset {fee_data.asset} should already be taken"
                         f" into account in buy {buy_data.asset} or sell "
                         f"{sell_data.asset} asset.")

    if (fee_data.asset is not None and fee_data.asset != 'USD' and
            fee_data.amount != 0.0):
        reduce_fifo(form8949, abs(fee_data.amount), fee_data.asset, fifo[fee_data.asset],
                    fee_data.total, fee_data.tx_date)

def main():
    """Run the FIFO capital-gains pipeline on asset_tx.csv.

    This function produces an IRS Form 8949–style output file.
    It reads ../asset_tx.csv, keeping only the necessary columns:
    Tx Index, Tx Date, Asset, Amount (asset), Sell price ($),
    Buy price ($), Type.  The rows from the CSV are then parsed
    and the FIFO info is updated, writing all sales to
    ../form8949_output.csv.
    """
    # Load your file from the project root folder
    input_file_path = "../asset_tx.csv"
    output_file_path = "../form8949_output.csv"
    df = pd.read_csv(input_file_path)

    # create Tx Date column with date format (instead of datetime) and
    # only keep pertinent columns
    df['Tx Date'] = pd.to_datetime(df['Date']).dt.date
    df = df[['Tx Index', 'Tx Date', 'Asset', 'Amount (asset)', 'Sell price ($)',
             'Buy price ($)', 'Type']]

    # Prepare FIFO ledger for each token
    fifo = defaultdict(deque)

    # Prepare output for Form 8949
    form8949 = []

    # Main loop
    idx = 0


    for idx, rows in df.groupby('Tx Index'):
        block_type = rows.iloc[0]['Type']

        if rows.empty:
            raise ValueError(f"No rows for Tx Index {idx}")


        """while idx <= max(df['Tx Index']):
            # define block
            rows = df.loc[df['Tx Index'] == idx]
            block_type = rows.iloc[0]['Type']"""

        # check that all transactions within block have same type
        if not (rows['Type'] == block_type).all():
            raise ValueError(f"Block does not have same type throughout. "
                             f"{rows}")

        # extract buy, sell, and fee info from rows
        buy_data, sell_data, fee_data = parse_row_data(block_type, rows)

        # update FIFO and form8949
        update_fifo(buy_data, sell_data, fee_data, form8949, fifo)

        idx += 1

    # Create .csv with output for f8949
    pd.DataFrame(form8949).to_csv(output_file_path, index=False)
    print("Success! Form 8949 data saved to "+output_file_path)

    # check against original.  Erase this later!!!
    if False:
        df_output = pd.DataFrame(form8949)
        df_original = pd.read_csv('../form8949_output_original.csv')
        pd.set_option('display.max_columns', None)  # Show all columns
        df_original['Proceeds difference'] = abs(df_original['Proceeds'].astype(float) - df_output['Proceeds'].astype(float))
        df_original['Cost Basis difference'] = abs(df_original['Cost Basis'].astype(float) - df_output['Cost Basis'].astype(float))
        print(df_original[df_original['Proceeds difference'] > 0.05])
        print(df_original[df_original['Cost Basis difference'] > 0.05])

if __name__ == "__main__":

    main()

