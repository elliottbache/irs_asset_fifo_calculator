"""
Calculate IRS capital gains taxes using FIFO method.

This module implements IRS-style FIFO cost-basis calculations for
stocks and other assets. Each purchase creates a "lot" stored in a
per-asset FIFO queue.  Each sale consumes lots from oldest to newest,
allocating cost basis and proceeds proportionally and emitting Form 8949
rows.

This program uses a CSV file as input.  This file is called
"asset_tx.csv" in the published example, but any name can be used,
using this name in the python call. Expected input CSV columns
(at minimum):

    ``Tx Index``, ``Date``, ``Asset``, ``Amount (asset)``,
    ``Sell price ($)``, ``Buy price ($)``, ``Type``

Additional columns such as ``Account number``, ``Entity``, ``Notes`` or
``Remaining`` may be present but are ignored by this module.

Key steps:

1. Group rows by ``Tx Index`` into logical transaction blocks.
2. For each block, classify buy/sell/fee rows and compute:
   buy data, sell data, and fee data.
3. Update per-asset FIFO queues and append realized sales to a
   Form 8949 list.
4. Write the Form 8949 list to "form8949_output.csv".

For a full worked example, see the FIFO overview section of the docs.
"""

import argparse
from collections import defaultdict, deque
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from math import isclose, isfinite
from typing import Any, Literal, TypedDict

import numpy as np
import pandas as pd

DEFAULT_INPUT_FILE = "asset_tx.csv"
DEFAULT_OUTPUT_FILE = "form8949.csv"


def parse_args(argv: Sequence[str]) -> tuple[str, str]:
    """Parse args from command line."""
    parser = argparse.ArgumentParser(
        prog="irs-fifo-taxes",
        description="FIFO-based IRS Form 8949 capital gains calculator for assets.",
    )

    parser.add_argument(
        "--input-file",
        default=DEFAULT_INPUT_FILE,
        help=f"list of transactions .csv (default: {DEFAULT_INPUT_FILE})",
    )
    parser.add_argument(
        "--output-file",
        default=DEFAULT_OUTPUT_FILE,
        help=f"Form8949 list of sales .csv (default: {DEFAULT_OUTPUT_FILE})",
    )

    ns = parser.parse_args(argv)

    return ns.input_file, ns.output_file


def _validate_sale_inputs(
    form8949: list[dict[str, str]],
    asset: str,
    amount: float,
    proceeds: float,
    cost_basis: float,
    acquisition_date: date,
    sale_date: date,
) -> None:
    """Validate the sale inputs."""
    if not isinstance(acquisition_date, date):
        raise TypeError(
            f"Acquisition date must be in date format.\n"
            f"{amount!s} {asset} purchase on {acquisition_date} is invalid."
        )
    if not isinstance(sale_date, date):
        raise TypeError(
            f"Sale date must be in date format.\n"
            f"{amount} {asset} sale on {sale_date} is invalid."
        )

    for name, value in (
        ("amount", amount),
        ("proceeds", proceeds),
        ("cost_basis", cost_basis),
    ):
        if not is_finite_number(value):
            raise TypeError(
                f"{name} is not a valid number: {value}."
                f" sale_date: {sale_date} asset: {asset} "
                f"amount: {amount}"
            )

    if amount < 0:
        raise ValueError(
            f"Amount must be greater than zero.\n"
            f"{amount} {asset} sale on {sale_date} "
            f"is negative."
        )

    if cost_basis < 0:
        raise ValueError(
            f"Cost basis must be greater than zero.\n{amount} "
            f"{asset} purchase on {acquisition_date} "
            f"is negative."
        )

    if not isinstance(form8949, list):
        raise TypeError("A list object must be passed. Create form8949 list first.")

    if acquisition_date > sale_date:
        raise ValueError(
            "Acquisition date must be before sale date.\n"
            + str(amount)
            + " "
            + asset
            + " sale on "
            + str(sale_date)
            + " is invalid."
        )


def record_sale(
    form8949: list[dict[str, str]],
    asset: str,
    amount: float,
    proceeds: float,
    cost_basis: float,
    acquisition_date: date,
    sale_date: date,
) -> None:
    """Record a sale.

    This takes various data about the sale and appends the data to the
    open Form 8949 file object.

    Args:
        form8949 (list[dict[str, str]]): Form 8949 list of dicts
         holding txs.
        asset (str): The asset name.
        amount (float): The amount of the asset units.
        proceeds (float): The gross dollar proceeds from this portion
         of the sale.
        cost_basis (float): The dollar cost basis for this portion of
         the asset, including purchase fees.
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

    _validate_sale_inputs(
        form8949, asset, amount, proceeds, cost_basis, acquisition_date, sale_date
    )

    if proceeds >= 0.005 or cost_basis >= 0.005:

        # place negative numbers in parentheses
        if proceeds - cost_basis < 0:
            gain_or_loss = f"({round(abs(proceeds - cost_basis),2):.2f})"
        else:
            gain_or_loss = f"{round(proceeds - cost_basis,2):.2f}"

        form8949.append(
            {
                "Description": f"{round(amount,8):.8f}" + " " + asset,
                "Date Acquired": acquisition_date.strftime("%m/%d/%Y"),
                "Date Sold": sale_date.strftime("%m/%d/%Y"),
                "Proceeds": f"{round(proceeds,2):.2f}",
                "Cost Basis": f"{round(cost_basis,2):.2f}",
                "Gain or Loss": gain_or_loss,
            }
        )


def is_finite_number(x: object) -> bool:
    """Return ``True`` if ``x`` is a finite (non-NaN, non-infinite, non-bool)
    real number."""
    return (
        isinstance(x, (int, float)) and not isinstance(x, bool) and isfinite(float(x))
    )


class FifoLot(TypedDict):
    """Single FIFO lot for an asset.

    Attributes:
        amount (float): Remaining asset quantity.
        price (float): Unit price in USD.
        cost (float): Total cost basis in USD.
        tx_date (date): Acquisition date of this lot.
    """

    amount: float
    price: float
    cost: float
    tx_date: date


def _validate_lot_structure(lot: FifoLot) -> None:
    """Ensure the lot has required keys and valid types/values."""
    # check if all necessary keys are present in fifo row
    required_keys = ["amount", "price", "cost", "tx_date"]
    if not all(key in lot for key in required_keys):
        raise KeyError(f"FIFO contains an invalid buy. {lot}")

    if not isinstance(lot["tx_date"], date):
        raise TypeError(f"FIFO contains an invalid buy date: {lot}.")

    for name, value in (
        ("amount", lot["amount"]),
        ("price", lot["price"]),
        ("cost", lot["cost"]),
    ):
        if not is_finite_number(value):
            raise TypeError(f"{name} is not a valid number: {value}.")

    if lot["amount"] < 0:
        raise ValueError(f"FIFO amount is negative for sale: {lot}.")

    if lot["cost"] < 0:
        raise ValueError(f"FIFO cost is negative for sale: {lot}.")


def reduce_fifo(
    form8949: list[dict[str, str]],
    sell_amount: float,
    asset: str,
    fifo_asset: deque[FifoLot],
    proceeds: float,
    sale_date: date,
) -> None:
    """Update FIFO lots for a sale.

    This is where the FIFO cost-basis math happens. Given a sale of
    ``sell_amount`` units of ``asset``, the function walks the existing
    lots in ``fifo_asset`` from oldest to newest, consuming quantities
    until the sale is fully matched. For each lot (or partial lot) that
    is used, it:

    1. Computes the fraction of the lot that is being sold.
    2. Allocates the same fraction of the lot's cost to this sale.
    3. Allocates a proportional share of the sale's total proceeds
       based on how many units from the lot are used.
    4. Records a Form 8949 row via :func:`record_sale`.
    5. Updates or removes the lot from ``fifo_asset`` to reflect what
       remains after the sale.

    In other words, the earliest purchases (oldest lots) are always
    matched first, which is exactly the FIFO (First In, First Out)
    method required for these cost-basis calculations.

    Example:
        Suppose you bought 10 NVDA at $10 (cost $100) and later
        5 NVDA at $11 (cost $55), then sell 12 NVDA for total proceeds
        of $144. The FIFO matching is:

        * First 10 units come from the $10 lot
        * Remaining 2 units come from the $11 lot

        This function will:
        * Create one Form 8949 row for the 10-unit slice of the first lot
        * Create another row for the 2-unit slice of the second lot
        * Leave 3 units in the second lot in ``fifo_asset``

    Args:
        form8949 (list[dict[str, str]]): Form 8949 list of dicts
            holding txs.
        sell_amount (float): this sale's amount
        asset (str): this asset
        fifo_asset (deque[FifoLot]):
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

    if sell_amount <= 0:
        raise ValueError(f"sell_amount must be positive, got {sell_amount}")

    amount_tol = 5e-9  # tolerance for remaining asset amount
    proceeds_tol = 5e-3  # tolerance for remaining asset proceeds
    remaining = sell_amount
    while remaining > amount_tol and fifo_asset:

        # set the current lot
        lot = fifo_asset[0]

        _validate_lot_structure(lot)

        if lot["amount"] == 0:
            fifo_asset.popleft()
            continue

        acquisition_date = lot["tx_date"]
        used = min(remaining, lot["amount"])

        # proportional cost and proceeds from used
        this_cost = used / lot["amount"] * lot["cost"]

        this_proceeds = used / sell_amount * proceeds

        record_sale(
            form8949, asset, used, this_proceeds, this_cost, acquisition_date, sale_date
        )

        lot["amount"] -= used
        if lot["amount"] == 0:
            fifo_asset.popleft()

        lot["cost"] -= this_cost

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

    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    if isinstance(value, str):
        clean_value = "".join(value.replace(",", "").split())
        try:
            return float(clean_value)
        except ValueError as e:
            raise ValueError(f"Invalid amount {value}: {e}") from e

    raise TypeError(f"Invalid amount {value}")


def is_fee(asset: str | None) -> bool:
    """Check if ``asset`` is a fee transaction.

    In order to be a fee, the asset must start with the letters "fee",
    and be longer than 3 characters.
    """
    return asset is not None and asset.startswith("fee") and len(asset) > 3


@dataclass
class AssetData:
    """Single transaction data row for a particular asset.

    Attributes:
        asset (str): The asset from the transaction.
        amount (float): The amount of the asset in this transaction.
        price (float): Unit price in USD.
        total (float): Total amount.  This can be cost or proceeds and
            is typically the ``amount * price
            +- fee_amount * fee_price``
        tx_date (date): Date of the transaction.
    """

    asset: str | None
    amount: float
    price: float
    total: float
    tx_date: date


BlockType = Literal["Buy", "Sell", "Exchange", "Transfer"]


def _find_trade_row_index(
    is_buy: bool, rows: pd.DataFrame, fee_rows: list[int]
) -> int | None:
    """Locate the index of the non-fee buy/sell row.

    Returns:
        The integer positional index into ``rows``, or None if no row
        is found.

    Raises:
        ValueError: If more than one matching non-fee row is found.
    """
    trade_idx = None
    for idx in range(len(rows)):
        if idx in fee_rows:
            continue

        amount = parse_amount(rows.iloc[idx]["Amount (asset)"])
        if (not is_buy and amount < 0) or (is_buy and amount > 0):
            if trade_idx is not None:
                raise ValueError(
                    f"Multiple rows for buy or sell must be" f" implemented {rows}."
                )
            trade_idx = idx

    return trade_idx


def _extract_trade_values(
    rows: pd.DataFrame, trade_idx: int | None, is_buy: bool
) -> tuple[str | None, float, float, float]:
    """Return (asset, amount, price) for the buy/sell side.

    If no trade row is found, returns (None, 0.0, 0.0, 0.0).
    """
    if trade_idx is None:
        return None, 0.0, 0.0, 0.0

    which_price = "Buy price ($)" if is_buy else "Sell price ($)"

    row = rows.iloc[trade_idx]
    asset = row["Asset"]
    amount = parse_amount(row["Amount (asset)"])  # negative
    price = parse_amount(row[which_price])

    # Dollar is always worth 1 dollar
    if asset == "USD":
        price = 1.0

    total = abs(amount * price)

    return asset, amount, price, total


def _adjust_amount_for_same_asset_fees(
    asset: str | None,
    amount: float,
    rows: pd.DataFrame,
    fee_assets: set[str],
    fee_rows: list[int],
) -> float:
    """Add same-asset fees to the trade amount (if applicable)."""
    # Add all fees in the buy or sell asset to the buy or sell amount
    if asset is None or asset not in fee_assets:
        return amount

    for idx in range(len(fee_rows)):
        row = rows.iloc[fee_rows[idx]]
        if row["Asset"] == "fee" + asset:
            amount += parse_amount(row["Amount (asset)"])

    return amount


def _apply_fee_rows_to_cost_or_proceeds(
    is_buy: bool,
    asset: str | None,
    price: float,
    total: float,
    rows: pd.DataFrame,
    fee_rows: list[int],
) -> float:
    """Adjust cost/proceeds by all fee rows, validating same-asset prices."""
    for idx in range(len(fee_rows)):
        row = rows.iloc[fee_rows[idx]]
        fee_amount = abs(parse_amount(row["Amount (asset)"]))
        fee_price = parse_amount(row["Sell price ($)"])

        # make sure fee price is same as buy or sell price if the same asset
        if (
            asset is not None
            and row["Asset"] == "fee" + asset
            and not isclose(price, fee_price, rel_tol=1e-6)
        ):
            raise ValueError(
                f"Fee price does not match buy or sell price"
                f" for \n{row} \n\nin \n{rows}."
            )

        if is_buy:
            total += fee_amount * fee_price
        else:
            total -= fee_amount * fee_price

    return total


def parse_buy_and_sell(
    is_buy: bool,
    block_type: BlockType,
    rows: pd.DataFrame,
    fee_assets: set[str],
    fee_rows: list[int],
) -> tuple[str | None, float, float, float]:
    """Extract the buy or sell side from a block of related transactions.

    For non-transfer blocks, this scans the rows (excluding fee rows) to
    find the one representing the buy or sell side based on the sign of
    ``Amount (asset)`` (``is_buy = True``: ``amount > 0``,
    ``is_buy = False``: ``amount < 0``). It returns that row's asset
    symbol, signed amount, and unit price. If the resulting asset is
    in ``fee_assets``, any fee rows are added to the returned amount.
    The price for USD is always forced to 1.0.

    Args:
        is_buy (bool): are we parsing buy side?
        block_type (BlockType): Type of block (``"Exchange"``,
            ``"Buy"``, ``"Sell"``, or ``"Transfer"``)
        rows (pd.DataFrame): the transactions for this block. Must
            include at least ``"Asset"``, ``"Amount (asset)"``,
            ``"Buy price ($)"``, and ``"Sell price ($)"``
        fee_assets (set[str]): assets that have associated fee rows
            (e.g. ``"USD"``, not ``"feeUSD"``)
        fee_rows (list[int]): indices within rows that correspond to fee
            transactions

    Returns:
        tuple[str, float, float, float]:
            A 4-tuple containing:

            - asset: the buy or sell asset
            - amount: the buy or sell amount
            - price: the buy or sell price
            - cost_or_proceeds: the cost if buy or proceeds if sell

            where

            * amount keeps the sign of the transaction (positive for buys,
              negative for sells).
            * price is the unit price in USD (forced to 1.0 for USD).
            * cost_or_proceeds is:

              - total cost (positive) for buys, including all relevant fees.
              - total proceeds (positive or negative with fee adjustments)
                for sells.

    Raises:
        ValueError: If more than one non-fee row matches the requested side.
        ValueError: If the fee asset is the same as the buy or sell asset
            but the prices are different.

    Notes:
        * The price for USD is always forced to ``1.0``.
        * Returns ``(None, 0.0, 0.0, 0.0)`` when:

          - ``block_type == "Transfer"``, or
          - no matching non-fee row is found.

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
        >>> fee_assets = set("USD")
        >>> parse_buy_and_sell(True, "Exchange", rows, fee_assets, fee_rows)
        ('NVDA', 10.0, 125.0, 1260.0)
        >>> parse_buy_and_sell(False, "Exchange", rows, fee_assets, fee_rows)
        ('TSLA', -25.0, 50.0, 1240.0)
    """

    if block_type == "Transfer":
        return None, 0.0, 0.0, 0.0

    trade_idx = _find_trade_row_index(is_buy, rows, fee_rows)

    trade_asset, trade_amount, trade_price, cost_or_proceeds = _extract_trade_values(
        rows, trade_idx, is_buy
    )

    trade_amount = _adjust_amount_for_same_asset_fees(
        trade_asset, trade_amount, rows, fee_assets, fee_rows
    )

    cost_or_proceeds = _apply_fee_rows_to_cost_or_proceeds(
        is_buy, trade_asset, trade_price, cost_or_proceeds, rows, fee_rows
    )

    return (trade_asset, trade_amount, trade_price, cost_or_proceeds)


def _prune_fee_rows_for_trade_assets(
    rows: pd.DataFrame,
    fee_rows: list[int],
    fee_assets: set[str],
    buy_asset: str | None,
    sell_asset: str | None,
) -> None:
    """Remove fee rows/assets that belong to the buy or sell asset.

    After this:
    - `fee_assets` contains only assets different from buy/sell.
    - `fee_rows` contains only rows whose underlying asset is not
      the buy or sell asset.
    """
    if buy_asset in fee_assets:
        fee_assets.remove(buy_asset)
    if sell_asset in fee_assets:
        fee_assets.remove(sell_asset)
    idx = 0
    while idx < len(fee_rows):
        if rows.iloc[fee_rows[idx]]["Asset"][len("fee") :] in [buy_asset, sell_asset]:
            del fee_rows[idx]
        else:
            idx += 1


def _aggregate_fee_asset(
    rows: pd.DataFrame,
    fee_rows: list[int],
    fee_assets: set[str],
) -> tuple[str | None, float, float]:
    """Aggregate remaining fee rows into (asset, amount, avg_price).

    Returns:
        (fee_asset, fee_amount, fee_price)

    Raises:
        ValueError: If more than one **different** fee asset remains.
        ValueError: If fees are positive.

    """
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
            this_amount = parse_amount(rows.iloc[fee_rows[idx]]["Amount (asset)"])
            fee_amount += this_amount
            fee_price += this_amount * parse_amount(
                rows.iloc[fee_rows[idx]]["Sell price ($)"]
            )
        if fee_amount == 0:
            fee_price = 0.0
        else:
            fee_price /= fee_amount

    if fee_amount > 0:
        raise ValueError(f"Fees cannot be positive: {fee_amount} for {rows}")

    return fee_asset, fee_amount, fee_price


def parse_row_data(
    block_type: BlockType, rows: pd.DataFrame
) -> tuple[AssetData, AssetData, AssetData]:
    """Extract the necessary values from row data.

    Args:
        block_type (BlockType): The type of block to extract from.  Can
            take the following values: [``"Buy"``, ``"Sell"``,
            ``"Exchange"``, ``"Transfer"``]
        rows (pd.DataFrame): The row data to extract from. The mandatory
            columns are: [``"Tx Index"``, ``"Tx Date"``, ``"Asset"``,
            ``"Amount (asset)"``, ``"Buy price ($)"``,
            ``"Sell price ($)"``, ``"Type"``]

    Returns:
        tuple[AssetData, AssetData, AssetData]:
            A 3-tuple with the following:

            - buy data
            - sell data
            - fee data

    Notes:
        - Transfer fees are not deducted although if paid with an asset,
          the conversion of the asset to USD is taxed.
        - For transfers, there is no buy data.  The fee data becomes the
          sell data.
        - Proceeds from fee assets are:

          * Added to cost if the fee asset is the same as the bought
            asset
          * Deducted from proceeds if the fee asset is the same as the
            sold asset
          * Recorded as a sale if it is a transfer or if they are
            different from both assets in a buy/sell/exchange.

        - Here we assume that there can only be a maximum of 1 fee asset
          besides the buy and sell assets. Sell and fee amount are
          negative in general.
        - If the fee asset is the same as the buy or sell asset, it is
          included in these, and the fee amount for that asset
          is set to 0.  If there are no other fee assets, then the fee
          asset will be None.
        - With large enough fees, the buy amount may become negative, in
          which case it will later be used to update FIFO (reduce and
          append to form8949) rather than append to FIFO.

    Example:
        >>> import pandas as pd
        >>> from datetime import date
        >>> from calculate_taxes import parse_row_data
        >>> block_type = 'Buy'
        >>> rows = pd.DataFrame({
        ...     'Tx Index': [0] * 3, 'Tx Date': [date(2024, 9, 4)] * 3,
        ...     'Asset': ['USD', 'NVDA', 'feeUSD'],
        ...     'Amount (asset)': [-1250, 10, -10],
        ...     'Sell price ($)': [1, 'NaN', 1],
        ...     'Buy price ($)': [1, 125, 1],
        ...     'Type': ['Buy'] * 3})
        >>> buy_data, sell_data, fee_data = parse_row_data(block_type, rows)
        >>> buy_data # doctest: +NORMALIZE_WHITESPACE
        AssetData(asset='NVDA', amount=10.0, price=125.0, total=1260.0,
        tx_date=datetime.date(2024, 9, 4))
        >>> sell_data # doctest: +NORMALIZE_WHITESPACE
        AssetData(asset='USD', amount=-1260.0, price=1.0, total=1240.0,
        tx_date=datetime.date(2024, 9, 4))
        >>> fee_data # doctest: +NORMALIZE_WHITESPACE
        AssetData(asset=None, amount=0.0, price=0.0, total=-0.0,
        tx_date=datetime.date(2024, 9, 4))
    """

    if block_type not in ["Buy", "Sell", "Exchange", "Transfer"]:
        raise ValueError(f"{block_type} is not a valid block type")

    # change to date format
    raw_date = rows.iloc[0]["Tx Date"]
    first_date = raw_date.date() if hasattr(raw_date, "date") else raw_date

    # identify fee rows and assets
    fee_rows = []
    fee_assets = set()
    for idx in range(len(rows)):
        if is_fee(rows.iloc[idx]["Asset"]):
            fee_rows.append(idx)
            fee_assets.add(rows.iloc[idx]["Asset"][len("fee") :])

    buy_asset, buy_amount, buy_price, cost = parse_buy_and_sell(
        True, block_type, rows, fee_assets, fee_rows
    )
    sell_asset, sell_amount, sell_price, proceeds = parse_buy_and_sell(
        False, block_type, rows, fee_assets, fee_rows
    )
    if buy_asset is not None and buy_asset == sell_asset:
        raise ValueError("Buy and sell asset cannot be the same.")

    _prune_fee_rows_for_trade_assets(rows, fee_rows, fee_assets, buy_asset, sell_asset)

    fee_asset, fee_amount, fee_price = _aggregate_fee_asset(rows, fee_rows, fee_assets)
    fee_proceeds = -fee_amount * fee_price  # positive

    # proceeds are 0 for transfers and purchases (USD doesn't give gains)
    if cost < 0:
        raise ValueError(f"Cost cannot be negative: {cost} for {rows}")

    buy_data = AssetData(
        asset=buy_asset,
        amount=float(buy_amount),
        price=float(buy_price),
        total=float(cost),
        tx_date=first_date,
    )
    sell_data = AssetData(
        asset=sell_asset,
        amount=float(sell_amount),
        price=float(sell_price),
        total=float(proceeds),
        tx_date=first_date,
    )
    fee_data = AssetData(
        asset=fee_asset,
        amount=float(fee_amount),
        price=float(fee_price),
        total=float(fee_proceeds),
        tx_date=first_date,
    )

    return buy_data, sell_data, fee_data


def update_fifo(
    buy_data: AssetData,
    sell_data: AssetData,
    fee_data: AssetData,
    form8949: list[dict[str, str]],
    fifo: defaultdict[str, deque[FifoLot]],
) -> None:
    """Updates FIFO dict of ``deque``  using info from this block of transactions.

    Args:
        buy_data (AssetData): buy info for this block of transactions
        sell_data (AssetData): sell info for this block of transactions
        fee_data (AssetData): fee info for this block of transactions
        form8949 (list[dict[str, str]]): Form 8949 list of dicts
         holding txs.
        fifo (defaultdict[str, deque[FifoLot]]):
            purchases of each token defined by their amount, price,
            cost, and date

    Returns:
        None

    Notes:
    - In general, buy and sell assets and fee asset should not be the
    same.  If they were that way upstream, the fees should have already
    been added to buy or sell and then set to 0.
    - If previously calculated fees are same asset as buy and larger than
    buy amount, the net buy amount is negative and is thus reduced from
    FIFO instead of appended.

    Example:
        Simple NVDA purchase that appends a new lot to the FIFO ledger:
        >>> from collections import defaultdict, deque
        >>> from datetime import date
        >>> from calculate_taxes import (AssetData, update_fifo)
        >>> fifo = defaultdict(deque)
        >>> form8949 = []
        >>> buy = AssetData(asset="NVDA", amount=10.0, price=100.0,
        ...                 total=1000.0, tx_date=date(2024, 1, 1))
        >>> sell = AssetData(asset="USD", amount=-1000.0, price=1.0,
        ...                  total=0.0, tx_date=date(2024, 1, 1))
        >>> fee = AssetData(asset=None, amount=0.0, price=0.0, total=0.0,
        ...                 tx_date=date(2024, 1, 1))
        >>> update_fifo(buy, sell, fee, form8949, fifo)
        >>> len(fifo["NVDA"])
        1
        >>> fifo["NVDA"][0]["amount"]
        10.0
        >>> fifo["NVDA"][0]["price"]
        100.0
        >>> fifo["NVDA"][0]["cost"]
        1000.0
        >>> fifo["NVDA"][0]["tx_date"]
        datetime.date(2024, 1, 1)
        >>> form8949
        []
    """

    if buy_data.asset is not None and buy_data.asset != "USD":
        if buy_data.amount > 0:
            fifo[buy_data.asset].append(
                {
                    "amount": buy_data.amount,
                    "price": buy_data.price,
                    "cost": buy_data.total,
                    "tx_date": buy_data.tx_date,
                }
            )
        elif buy_data.amount < 0:  # if fees exceed buy amount
            reduce_fifo(
                form8949,
                abs(buy_data.amount),
                buy_data.asset,
                fifo[buy_data.asset],
                buy_data.total,
                buy_data.tx_date,
            )

    if (
        sell_data.asset is not None
        and sell_data.asset != "USD"
        and sell_data.amount < 0
    ):
        reduce_fifo(
            form8949,
            abs(sell_data.amount),
            sell_data.asset,
            fifo[sell_data.asset],
            sell_data.total,
            sell_data.tx_date,
        )

    # if they are the same, the fees are already taken into account in
    # the buy_data and sell_data
    if fee_data.asset == buy_data.asset or fee_data.asset == sell_data.asset:
        raise ValueError(
            f"Fee asset {fee_data.asset} should already be taken"
            f" into account in buy {buy_data.asset} or sell "
            f"{sell_data.asset} asset."
        )

    if (
        fee_data.asset is not None
        and fee_data.asset != "USD"
        and fee_data.amount != 0.0
    ):
        reduce_fifo(
            form8949,
            abs(fee_data.amount),
            fee_data.asset,
            fifo[fee_data.asset],
            fee_data.total,
            fee_data.tx_date,
        )


def run_fifo_pipeline(df: pd.DataFrame) -> list[dict[str, str]]:
    """Run the FIFO capital-gains pipeline on a transactions DataFrame.

    The input DataFrame must contain at least the following columns:

    - ``"Date"``
    - ``"Tx Index"``
    - ``"Asset"``
    - ``"Amount (asset)"``
    - ``"Sell price ($)"``
    - ``"Buy price ($)"``
    - ``"Type"``

    This function is pure with respect to IO: it does not read or write
    any files. It returns a list of dictionaries representing rows for
    an IRS Form 8949-style output.

    Args:
        df (pd.DataFrame): Raw transaction DataFrame with the columns
            described above.

    Returns:
        list[dict[str, str]]:
        A list of Form 8949 rows (dicts with keys:
        ``"Description"``, ``"Date Acquired"``, ``"Date Sold"``,
        ``"Proceeds"``, ``"Cost Basis"``, ``"Gain or Loss"``).

    Example:
        >>> import pandas as pd
        >>> from calculate_taxes import run_fifo_pipeline
        >>> df = pd.DataFrame([
        ...     # buy block (Tx Index 0)
        ...     {"Date": "2024-09-04", "Tx Index": 0, "Asset": "USD",
        ...      "Amount (asset)": -1250.0, "Sell price ($)": 1.0,
        ...      "Buy price ($)": 1.0, "Type": "Buy"},
        ...     {"Date": "2024-09-04", "Tx Index": 0, "Asset": "NVDA",
        ...      "Amount (asset)": 10.0, "Sell price ($)": float('nan'),
        ...      "Buy price ($)": 125.0, "Type": "Buy"},
        ...     {"Date": "2024-09-04", "Tx Index": 0, "Asset": "feeUSD",
        ...      "Amount (asset)": -10.0, "Sell price ($)": 1.0,
        ...      "Buy price ($)": float('nan'), "Type": "Buy"},
        ...     # sell block (Tx Index 1)
        ...     {"Date": "2024-09-05", "Tx Index": 1, "Asset": "NVDA",
        ...      "Amount (asset)": -4.0, "Sell price ($)": 130.0,
        ...      "Buy price ($)": float('nan'), "Type": "Sell"},
        ...     {"Date": "2024-09-05", "Tx Index": 1, "Asset": "USD",
        ...      "Amount (asset)": 520.0, "Sell price ($)": float('nan'),
        ...      "Buy price ($)": 1.0, "Type": "Sell"},
        ...     {"Date": "2024-09-05", "Tx Index": 1, "Asset": "feeUSD",
        ...      "Amount (asset)": -1.0, "Sell price ($)": 1.0,
        ...      "Buy price ($)": float('nan'), "Type": "Sell"},
        ... ])
        >>> rows = run_fifo_pipeline(df)
        >>> len(rows) >= 1
        True
        >>> sorted(rows[0].keys())  # doctest: +NORMALIZE_WHITESPACE
        ['Cost Basis', 'Date Acquired', 'Date Sold', 'Description',
         'Gain or Loss', 'Proceeds']
        >>> any(r["Description"].endswith("NVDA") for r in rows)
        True
    """
    # Work on a copy so callers keep their original df unchanged
    df = df.copy()

    # create Tx Date column with date format (instead of datetime) and
    # only keep pertinent columns
    df["Tx Date"] = pd.to_datetime(df["Date"]).dt.date
    df = df[
        [
            "Tx Index",
            "Tx Date",
            "Asset",
            "Amount (asset)",
            "Sell price ($)",
            "Buy price ($)",
            "Type",
        ]
    ]

    # prepare FIFO ledger for each token
    fifo: defaultdict[str, deque[FifoLot]] = defaultdict(deque)

    # prepare output for Form 8949
    form8949: list[dict[str, str]] = []

    # main loop (pure pipeline logic)
    for idx, rows in df.groupby("Tx Index"):
        block_type = rows.iloc[0]["Type"]

        if rows.empty:
            raise ValueError(f"No rows for Tx Index {idx}")

        # check that all transactions within block have same type
        if not (rows["Type"] == block_type).all():
            raise ValueError("Block does not have same type throughout. " f"{rows}")

        # extract buy, sell, and fee info from rows
        buy_data, sell_data, fee_data = parse_row_data(block_type, rows)

        # update FIFO and form8949
        update_fifo(buy_data, sell_data, fee_data, form8949, fifo)

    return form8949


def main(argv: Sequence[str] | None = None) -> None:
    """Run the FIFO capital-gains pipeline on a CSV file.

    This function produces an IRS Form 8949-style output file.

    It reads input_file_path (by default "asset_tx.csv"),
    keeping only the necessary columns:

        ``"Tx Index"``, ``"Date"``, ``"Asset"``, ``"Amount (asset)"``,
        ``"Sell price ($)"``, ``"Buy price ($)"``, ``"Type"``

    It then parses the rows, updates the FIFO ledger, and writes all
    sales to output_file_path (by default "form8949_output.csv").

    Args:
        input_file_path: Path to the input CSV with raw transactions.
        output_file_path: Path where the Form 8949-style CSV will be
            written.

    Returns:
        None.
    """
    # parse args
    input_file, output_file = parse_args(argv)

    # IO-only wrapper around the pure pipeline
    df = pd.read_csv(input_file)
    form8949 = run_fifo_pipeline(df)

    pd.DataFrame(form8949).to_csv(output_file, index=False)
    print(f"Success! Form 8949 data saved to {output_file}")

    # check against original.  Erase this later!!!
    if False:
        df_output = pd.DataFrame(form8949)
        df_original = pd.read_csv("form8949_output_original.csv")
        pd.set_option("display.max_columns", None)  # Show all columns
        df_original["Proceeds difference"] = abs(
            df_original["Proceeds"].astype(float) - df_output["Proceeds"].astype(float)
        )
        df_original["Cost Basis difference"] = abs(
            df_original["Cost Basis"].astype(float)
            - df_output["Cost Basis"].astype(float)
        )
        print(df_original[df_original["Proceeds difference"] > 0.05])
        print(df_original[df_original["Cost Basis difference"] > 0.05])


if __name__ == "__main__":

    main()
