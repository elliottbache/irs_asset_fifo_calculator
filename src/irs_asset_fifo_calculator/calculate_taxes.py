"""
Calculate IRS capital gains taxes using FIFO method.

Tax calculator that tracks capital gains from multiple purchases and
sales.  This program uses a CSV file as input.  This file is called
"asset_tx.csv" in the published example, but any name can be used,
using this name in the python call. The file has the following header:
Date,Asset,Amount (asset), Sell price ($),Buy price ($),Account number,
Entity,Notes,Remaining

Transfer fees are not deducted although if paid with an asset, the
conversion of the asset to USD is taxed.

Functions:
    record_sale:
        Write a sale to the Form 8949 file object.

    main:
        Main function.

"""

from datetime import date
from typing import List, Dict, DefaultDict, Deque, TypedDict, Any, Literal
from collections import defaultdict, deque
import numbers
import pandas as pd
from dataclasses import dataclass

def record_sale(form8949: List[Dict[str, str]], asset: str, amount: float,
                proceeds: float, cost_basis: float, acquisition_date: date,
                sale_date: date) -> None:
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
        acquisition_date (date): The acquisition date.
        sale_date (date): The sale date.

    Returns:
        None.

    Example:
        >>> from calculate_taxes import record_sale
        >>> from datetime import date
        >>> form8949 = list()
        >>> form8949.append({"Description": "10.00000000 NVDA",
        ...     "Date Acquired": "1982-10-27",
        ...     "Date Sold": "2024-12-31",
        ...     "Proceeds": "10000",
        ...     "Cost Basis": "1000",
        ...     "Gain or Loss": "9000",
        ...     "Code": "",
        ...     "Adjustment Amount": ""})
        >>> record_sale(form8949, "TSLA", 10, 100, 90, date(2024,1,1),
        ...     date(2024,12,31))
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

    if not isinstance(acquisition_date, date):
        raise TypeError(
            "Acquisition date must be in date format.\n"
            + str(amount) + " " + asset + " purchase on "
            + acquisition_date + " is invalid."
        )
    if not isinstance(sale_date, date):
        raise TypeError(
            "Sale date must be in date format.\n",
            amount, " ", asset, "sale on ", sale_date, " is invalid."
        )

    if not ((isinstance(amount, float) or isinstance(amount, int)) and
            (isinstance(proceeds, float) or isinstance(proceeds, int)) and
            (isinstance(cost_basis, float) or isinstance(cost_basis, int))):
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

class FifoLot(TypedDict):
    amount: float
    price: float
    cost: float
    tx_date: date

def update_fifo(
        form8949: List[Dict[str, str]], sell_amount: float, asset: str,
        fifo: DefaultDict[str, Deque[FifoLot]],
        proceeds: float,
        sale_date: date) -> None:
    """Update FIFO lots for a sale.

    Takes a sale and reduces the FIFO lots for that asset by the sale amount,
    recording one or more rows in form8949.

    Args:
        form8949 (List[Dict[str, str]]): Form 8949 list of dicts
            holding txs.
        sell_amount (float): this sale's amount
        asset (str): this asset
        fifo (DefaultDict[str, Deque[FifoLot]]):
            purchases of each token defined by their amount, price,
            cost, and date
        proceeds (float): this sale's proceeds
        sale_date (date): this sale's date

    Returns:
        None

    Example:
        >>> from calculate_taxes import update_fifo
        >>> from datetime import date
        >>> from collections import defaultdict, deque
        >>> form8949 = list()
        >>> fifo = defaultdict(deque)
        >>> fifo['NVDA'].append({"amount": 10, "price": 10,
        ...     "cost": 100*1.002, "tx_date": date(2024, 1, 1)})
        >>> fifo['NVDA'].append({"amount": 20, "price": 11,
        ...     "cost": 210*1.002, "tx_date": date(2024, 2, 1)})
        >>> update_fifo(form8949, 15, 'NVDA', fifo, 135,
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

    if asset not in fifo or not fifo[asset]:
        raise ValueError(f"Fifo does not contain {asset}.")

    if sell_amount < 0:
        sell_amount = abs(sell_amount)

    remaining = sell_amount
    while remaining > 0 and fifo[asset]:

        # set the current lot
        lot = fifo[asset][0]

        # check if all necessary keys are present in fifo row
        required_keys = ['amount', 'price', 'cost', 'tx_date']
        if not all(key in lot for key in required_keys):
            raise KeyError(f"FIFO contains an invalid purchase. {lot}")

        # check if all data is the right type
        if not isinstance(lot['amount'], numbers.Number) \
                or not isinstance(lot['price'], numbers.Number) \
                or not isinstance(lot['cost'], numbers.Number) \
                or not isinstance(lot['tx_date'], date):
            raise TypeError(f"FIFO contains an invalid purchase. {lot}")

        if lot['amount'] == 0:
            fifo[asset].popleft()
            continue
        elif lot['amount'] < 0:
            lot['amount'] = abs(lot['amount'])

        if lot['cost'] < 0:
            lot['cost'] = abs(lot['cost'])

        acquisition_date = lot['tx_date']
        used = min(remaining, lot['amount'])

        # proportional cost and proceeds from used
        this_cost = used / lot['amount'] * lot['cost']

        this_proceeds = used / sell_amount * proceeds

        record_sale(form8949, asset, used, this_proceeds, this_cost,
                    acquisition_date, sale_date)

        lot['amount'] -= used
        if lot['amount'] == 0:
            fifo[asset].popleft()

        lot['cost'] -= this_cost

        remaining -= used

def parse_amount(value: Any) -> float:
    """Parse amount from input.  Can be string or numeric."""

    if isinstance(value, numbers.Number):
        return float(value)

    if isinstance(value, str):
        clean_value = "".join(value.replace(',', '').split())
        try:
            return float(clean_value)
        except ValueError:
            raise ValueError(f"Invalid amount {value}")

    raise TypeError(f"Invalid amount {value}")

def define_amounts(row0: pd.Series, row1: pd.Series) -> tuple[float, float]:
    """Define amount0 and amount1.

    These amounts correspond to the asset being sold and the asset being
    bought.  In most situations, one of the assets will be USD.

    Args:
        row0 (pd.Series): row containing the first assets info
        row1 (pd.Series): row containing the second assets info

    Returns:
        tuple[float, float]: amount0, amount1

    Example:
        >>> from calculate_taxes import define_amounts
        >>> import pandas as pd
        >>> from datetime import date
        >>> row0 = pd.Series({'Tx Date': date(2024, 9, 4), 'Asset': 'USD',
        ...       'Amount (asset)': -1250.0, 'Sell price ($)': 1.0,
        ...       'Buy price ($)': 1.0, 'Account number': 1234})
        >>> row1 = pd.Series({'Tx Date': date(2024,9,4), 'Asset': 'NVDA',
        ...        'Amount (asset)': 10.0, 'Sell price ($)': 'NaN',
        ...        'Buy price ($)': 12.0, 'Account number': 1234})
        >>> define_amounts(row0, row1)
        (-1250.0, 10.0)
    """

    return parse_amount(row0['Amount (asset)']), parse_amount(row1['Amount (asset)'])

def define_blocks(row0: pd.Series, row1: pd.Series) -> tuple[str, int]:
    """Define blocks based on two rows of related transactions.

    Possible block_types = approved_exchange (requiring an extra approval transaction),
        transfer, purchase, sale, exchange (where USD is not involved)

    Args:
        row0 (pd.Series): row containing the first assets info
        row1 (pd.Series): row containing the second assets info

    Returns:
        tuple[str, int]: block type, number of transactions in this block

    Example:
        >>> from calculate_taxes import define_blocks
        >>> import pandas as pd
        >>> from datetime import date
        >>> row0 = pd.Series({'Tx Date': date(2024, 9, 4), 'Asset': 'USD',
        ...       'Amount (asset)': -1250.0, 'Sell price ($)': 1.0,
        ...       'Buy price ($)': 1.0, 'Account number': 1234})
        >>> row1 = pd.Series({'Tx Date': date(2024, 9, 4), 'Asset': 'NVDA',
        ...        'Amount (asset)': 10.0, 'Sell price ($)': 'NaN',
        ...        'Buy price ($)': 12.0, 'Account number': 1234})
        >>> define_blocks(row0, row1)
        ('purchase', 3)
    """

    asset0 = str(row0['Asset'])
    asset1 = str(row1['Asset'])
    account0 = row0['Account number']

    amount0, amount1 = define_amounts(row0, row1)

    # special case: Exchange with separate approval + fee row
    if account0 == 'Approved' and asset0.startswith("fee") and len(asset0) > 3:
        block_type = 'approved_exchange'
        n_tx = 4
    # TEMP: treat account IDs with '-' as transfers
    elif '-' in str(account0):
        block_type = 'transfer'
        n_tx = 2
    # Fiat → asset
    elif asset0 == 'USD' and amount1 > 0:
        block_type = 'purchase'
        n_tx = 3
    # Asset → fiat
    elif asset1 == 'USD' and amount0 < 0:
        block_type = 'sale'
        n_tx = 3
    # Asset → asset
    elif (asset0 != 'USD' and asset1 != 'USD'
          and amount0 < 0 and amount1 > 0):
        block_type = 'exchange'
        n_tx = 3
    else:
        raise ValueError(f"Invalid block: could not classify transaction pair"
                         "\naccount0: {account0}\nasset0: {asset0} amount0: {amount0}"
                         "\nasset1: {asset1} amount1: {amount1}")

    return block_type, n_tx

def is_fee(asset: str) -> bool:
    """Check if asset is a fee transaction.

    In order to be a fee, the asset must start with the letters 'fee',
    and be longer than 5 characters.  This will help weed out assets
    that begin with the letters fee (improbable) and do not have at
    least another 3 letters as tickers usually have 3 or 4 letters.
    """
    if asset.startswith("fee") and len(asset) > 5:
        return True
    else:
        return False

def check_fees(block_type: str, rows: pd.DataFrame) -> None:
    """Check dataframe for fees in the correct rows.

    Args:
        block_type (str): type of block
        rows (pd.DataFrame): dataframe containing transactions for this block

    Returns:
        None

    Raises:
        ValueError: if the rows do not contain the right number of fee
         entries given by the block_type.

    Example:
        >>> from calculate_taxes import check_fees
        >>> import pandas as pd
        >>> block_type = 'purchase'
        >>> rows = pd.DataFrame({'Tx Date': [date(2024, 9, 4)] * 3,
        ...     'Asset': ['USD', 'NVDA', 'feeUSD'],
        ...     'Amount (asset)': [-1250, 10, -5],
        ...     'Sell price ($)': [1, 'NaN', 1],
        ...     'Buy price ($)': [1, 125, 1],
        ...     'Account number': [1234, 1234, 1234]
        ...     })
        >>> check_fees(block_type, rows)

        """

    if len(rows) == 0:
        raise ValueError('Empty dataframe.')

    first_asset = rows.iloc[0]['Asset']
    last_asset = rows.iloc[-1]['Asset']

    message_missing = (f"Invalid block: missing fee for "
                + f"{rows.iloc[0]['Amount (asset)']} {first_asset}"
                + f" on {rows.iloc[0]['Tx Date']}"
    )
    message_extra = (f"Invalid block: extra fee for "
                + f"{rows.iloc[0]['Amount (asset)']} {first_asset}"
                + f" on {rows.iloc[0]['Tx Date']}"
    )
    message_approval = (f"Invalid block: missing approval fee for "
                + f"{rows.iloc[0]['Amount (asset)']} {first_asset}"
                + f" on {rows.iloc[0]['Tx Date']}"
    )

    if block_type == 'transfer':
        if is_fee(first_asset):
            raise ValueError(message_extra)
        if not is_fee(last_asset):
            raise ValueError(message_missing)
        return None

    if block_type == 'approved_exchange':
        if not is_fee(first_asset):
            raise ValueError(message_approval)
        if not is_fee(last_asset):
            raise ValueError(message_missing)
        if is_fee(rows.iloc[1]['Asset']):
            raise ValueError(message_extra)
        if is_fee(rows.iloc[2]['Asset']):
            raise ValueError(message_extra)
        return None

    # all other block types have 3 tx
    if is_fee(first_asset):
        raise ValueError(message_extra)
    if not is_fee(last_asset):
        raise ValueError(message_missing)
    if is_fee(rows.iloc[1]['Asset']):
        raise ValueError(message_extra)


@dataclass
class AssetData:
    asset: str
    amount: float
    price: float
    total: float
    tx_date: date


BlockType = Literal['purchase', 'sale', 'approved_exchange', 'exchange', 'transfer']


def parse_row_data(block_type: BlockType, rows: pd.DataFrame) -> tuple[AssetData, AssetData, AssetData]:
    """Extract the necessary values from row data.

    Notes:
    - Transfer fees are not deducted although if paid with an asset, the
        conversion of the asset to USD is taxed.
    - Proceeds from fee assets are:
        - Added to cost if the fee asset is the same as the bought asset
        - Deducted from proceeds if the fee asset is the same as the sold asset
        - Recorded as a sale if it is a transfer or if they are different from
            both assets in a purchase/sale/exchange.

    Args:
        block_type (BlockType): The type of block to extract from.  Can
            take the following values: ['purchase', 'sale',
            'approved_exchange', 'exchange', 'transfer']
        rows (pd.DataFrame): The row data to extract from. The mandatory
            columns are: [Tx Date, Asset, Amount (asset), Buy price ($),
            Sell price ($)]

    Returns:
        AssetData, AssetData, AssetData: buy data, sell data, and fee
            data.  Sell and fee amount are negative in general. Buy
            amount may become negative, in which case it will later
            be used to update FIFO (reduce and append to form8949)
            rather than append to FIFO.  For transfers, buy and sell
            amount and price are 0.  Buy asset is the transfer asset,
            sell asset is the fee asset.


    Example:
        >>> import pandas as pd
        >>> from datetime import date
        >>> from calculate_taxes import parse_row_data
        >>> block_type = 'purchase'
        >>> rows = pd.DataFrame({'Tx Date': [date(2024, 9, 4)] * 3,
        ...     'Asset': ['USD', 'NVDA', 'feeUSD'],
        ...     'Amount (asset)': [-1250, 10, -10],
        ...     'Sell price ($)': [1, 'NaN', 1],
        ...     'Buy price ($)': [1, 125, 1],
        ...     'Account number': [1234, 1234, 1234]})
        >>> buy_data, sell_data, fee_data = parse_row_data(block_type, rows)
        >>> buy_data
        AssetData(asset='NVDA', amount=10.0, price=125.0, total=1260.0, tx_date=datetime.date(2024, 9, 4))
        >>> sell_data
        AssetData(asset='USD', amount=-1260.0, price=1.0, total=0.0, tx_date=datetime.date(2024, 9, 4))
        >>> fee_data
        AssetData(asset='USD', amount=-10.0, price=1.0, total=10.0, tx_date=datetime.date(2024, 9, 4))
    """

    if block_type not in ['purchase', 'sale', 'approved_exchange', 'exchange', 'transfer']:
        raise ValueError(f"{block_type} is not a valid block type")

    expected_rows = {'purchase': 3, 'sale': 3, 'approved_exchange': 4, 'exchange': 3, 'transfer': 2}
    if expected_rows[block_type] != len(rows):
        raise ValueError(f"Incorrect number of rows for {block_type} ({len(rows)}) at {rows.iloc[0]['Tx Date']}")

    # change to date format
    raw_date = rows.iloc[0]['Tx Date']
    if hasattr(raw_date, 'date'):
        first_date = raw_date.date()
    else:
        first_date = raw_date

    # parse final row fee data
    fee_element = rows.iloc[-1]['Asset']
    if not is_fee(fee_element):
        raise ValueError(f"{fee_element} is not a valid fee element at {first_date}")
    fee_asset = fee_element[len("fee"):]
    fee_amount = parse_amount(rows.iloc[-1]['Amount (asset)'])

    # for approval exchange, parse first row fee data
    if block_type == 'approved_exchange':
        fee_element = rows.iloc[0]['Asset']
        if not is_fee(fee_element):
            raise ValueError(f"{fee_element} is not a valid fee element at {first_date}")

        # if fee assets are different
        if fee_asset != fee_element[len("fee"):]:
            raise ValueError(f'Fee asset mismatch. {fee_asset} is not the same as {fee_element[len("fee"):]} at {first_date}')

        # approval fee is added to exchange fee
        fee_amount += parse_amount(rows.iloc[0]['Amount (asset)'])

    # the price of a dollar should always be 1 dollar
    if fee_asset == 'USD':
        fee_price = 1.0
    else:
        fee_price = parse_amount(rows.iloc[-1]['Sell price ($)'])

    # identify row with sell data
    if block_type == 'approved_exchange':
        sell_idx = 1
    elif block_type == 'transfer':
        sell_idx = None
    else:
        sell_idx = 0

    # define sell data
    if sell_idx is None:
        sell_amount = 0.0
        sell_price = 0.0
        sell_asset = None
    else:
        sell_amount = parse_amount(rows.iloc[sell_idx]['Amount (asset)'])
        sell_price = parse_amount(rows.iloc[sell_idx]['Sell price ($)'])
        sell_asset = rows.iloc[sell_idx]['Asset']

    # identify row with buy data
    if block_type in ['purchase', 'exchange', 'sale']:
        buy_idx = 1
    elif block_type == 'approved_exchange':
        buy_idx = 2
    else:
        buy_idx = None

    # define buy data
    if buy_idx is None:
        buy_asset = None
        buy_amount = 0.0
        buy_price = 0.0
    else:
        buy_asset = rows.iloc[buy_idx]['Asset']
        buy_amount = parse_amount(rows.iloc[buy_idx]['Amount (asset)'])
        buy_price = parse_amount(rows.iloc[buy_idx]['Buy price ($)'])

    # define proceeds using USD amounts when available, otherwise use sell data
    # proceeds are 0 for transfers and purchases (USD doesn't give gains)
    if block_type == 'sale':
        proceeds = abs(buy_amount * buy_price) - abs(fee_amount * fee_price)
    elif block_type in ['exchange', 'approved_exchange']:
        proceeds = abs(sell_amount * sell_price) - abs(fee_amount * fee_price)
    else:
        proceeds = 0

    # define cost with using USD amounts when available, otherwise use buy data
    # proceeds are 0 for transfers and purchases (USD doesn't give gains)
    if block_type in ['purchase', 'exchange', 'approved_exchange']:
        cost = abs(sell_amount * sell_price) + abs(fee_amount * fee_price)
    else:
        cost = 0

    # transfer asset amounts are immutable
    if block_type != 'transfer':
        if fee_asset == buy_asset:
            buy_amount -= abs(fee_amount)
        if fee_asset == sell_asset:
            sell_amount -= abs(fee_amount)

    buy_data = AssetData(asset = buy_asset, amount = float(buy_amount), price=float(buy_price), total=float(cost), tx_date=first_date)
    sell_data = AssetData(asset=sell_asset, amount=float(sell_amount), price=float(sell_price), total=float(proceeds), tx_date=first_date)
    fee_data = AssetData(asset=fee_asset, amount=float(fee_amount), price=float(fee_price), total=abs(float(fee_amount*fee_price)), tx_date=first_date)

    return buy_data, sell_data, fee_data


    """if block_type in ['purchase', 'exchange', 'approved_exchange']:
#        buy_lot: FifoLot = {"amount": buy_amount, "price": buy_price, "cost": cost, "timestamp": timestamp}
        fifo[buy_asset].append({"amount": buy_amount, "price": buy_price, "cost": cost, "timestamp": timestamp})

    # ERRROR HANDLING IF THESE BECOME NEGATIVE!!

    if block_type in ['sale', 'exchange', 'approved_exchange']:
        update_fifo(form8949, abs(sell_amount), sell_asset, fifo, proceeds, timestamp)

    # if fee_token is not one of the exchanged tokens, update FIFO
    if block_type == 'transfer' or (fee_asset != sell_asset and fee_asset != buy_asset and fee_asset != 'USD'):
        sell_amount = fee_amount
        proceeds = abs(fee_amount * fee_price)
        update_fifo(form8949, abs(sell_amount), fee_asset, fifo, proceeds, timestamp)"""

if __name__ == "__main__":
    """
    asset = 'AMZN'
    amount = 10
    proceeds = 100
    price = 10
    cost_basis = 100
    acquisition_date = datetime(2024, 1, 1)
    sale_date = datetime.now()

    try:
        record_sale(form8949, asset, amount, proceeds, cost_basis,
                    acquisition_date, sale_date)
    except (KeyError, TypeError) as err:
        print(f"Error recording sale: {err}")
        sys.exit(1)

    print(form8949)

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

    try:
        update_fifo(form8949, 5, asset, fifo, 60, datetime(2024, 4, 1))
    except (KeyError, TypeError) as err:
        print(f"{type(err)} Error updating FIFO: {err}")
        sys.exit(1)
    print(fifo)

    try:
        update_fifo(form8949, 6, asset, fifo, 70, datetime(2024, 4, 2))
    except (KeyError, TypeError) as err:
        print(f"{type(err)} Error updating FIFO: {err}")
        sys.exit(1)
    print(fifo)
    """


    # Load your file from the project root folder
    file_path = "../asset_tx.csv"
    df = pd.read_csv(file_path)
    df['Tx Date'] = pd.to_datetime(df['Date']).dt.date
    df = df[['Tx Date', 'Asset', 'Amount (asset)', 'Sell price ($)', 'Buy price ($)', 'Account number']]

    pd.set_option('display.max_rows', None)  # Show all rows
    pd.set_option('display.max_columns', None)  # Show all columns
    print(f"df: {df}")

    # Prepare FIFO ledger for each token
    fifo = defaultdict(deque)

    # Output for Form 8949
    form8949 = []

    # Main loop
    len_df = len(df)
    idx = 0
    while idx < len_df:
        # print(f"Processing row {idx}: {df.iloc[idx]}")
        if idx < len_df - 1:
            amount0, amount1 = define_amounts(df.iloc[idx], df.iloc[idx + 1])
        else:
            print("1-block transactions must be implemented.")
        # print(f"Amounts {amount0}, {amount1}")
        idx += 1


    print("Only pass fifo[asset] to update_fifo.")
    print("Identify approved, transfer, etc in csv")

