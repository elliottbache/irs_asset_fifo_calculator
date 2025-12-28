# How the FIFO capital gains calculation works

This page explains the concepts behind the **FIFO capital-gains engine**
implemented in `irs_asset_fifo_calculator`.  
It is meant as a human-readable overview of what the code is doing, without
diving into implementation details.

---

## Overview

The goal of this project is to compute **IRS Form 8949**-style rows for trades
in stocks or other assets, using a **First In, First Out (FIFO)**
cost-basis method.

You start with a CSV of raw transactions (`asset_tx.csv`). The library:

1. Groups low-level CSV rows into **logical transaction blocks** using
   `Tx Index` and `Type` (Buy, Sell, Exchange, Transfer).
2. For each block, extracts three conceptual pieces:
   - **Buy side** (`buy_data`)
   - **Sell side** (`sell_data`)
   - **Fee side** (`fee_data`)
3. Maintains a **FIFO queue of lots** for each asset.
4. On sales (and some fees), it **consumes** those lots in order and writes
   **Form 8949 rows** describing each realized gain or loss.

The entry point for this logic is:

- `run_fifo_pipeline(df: pd.DataFrame) -> list[dict[str, str]]`  
  (pure data-in / data-out)
- `main(...)` in the CLI wrapper just handles reading/writing CSV and calls
  `run_fifo_pipeline`.

---

## FIFO logic

### Buy, Sell, or Exchange
Every time a purchase, sale, or exchange is made, the holdings of one asset 
will probably increase, and the holdings of another asset will decrease.  The
changes in these assets must be documented, calculating proceeds minus cost
basis for tax calculation purposes.  Purchases are added to the FIFO ledger,
and sales are deducted from the ledger starting with the oldest lots.  The only 
exception to these rules are for USD, which is not tracked since it does not generate gains
or losses.

Fees may be incurred for these transactions and can be deducted.
Three cases may occur:
- If the fee asset is the same as the bought asset, then the fee amount is directly 
deducted from the amount of the bought asset that is added to the FIFO ledger.
This deduction also affects the cost basis of the bought asset, increasing it.  
- If the fee asset is the same as the sold asset, then it is directly 
added to the amount of the sold asset that is used to update the FIFO ledger.
This deduction also affects the proceeds of the sold asset, decreasing it.
- If the fee asset is different from both the buy and sell assets, then
it is treated as a sale, and the FIFO ledger is updated accordingly, reducing
the oldest lot for that asset.

Fees for each transaction can be deducted in the buy asset, the sell asset, and in 
one more asset.  Two fee assets that are neither the buy nor sell asset will result 
in an error.

### Transfer
For transfers, the fees are not deducted, but are taken into account when
updated the FIFO ledger, where they are subtracted from the oldest lot
for that asset.

## Lots and the FIFO ledger

For each asset (e.g. `NVDA` or `TSLA`), the calculator keeps a FIFO queue of
**lots**:

```python
FifoLot:
    amount: float   # units of the asset
    price: float    # unit price in USD
    cost: float     # total cost in USD (including fees)
    tx_date: date   # acquisition date
```

Conceptually:
- A buy creates a new lot and appends it to the right end of the queue.
- A sell takes units from the left (oldest lot first) and may consume
multiple lots.
- After a sale, the remaining amount and cost in each lot are updated.

### Example
2 purchases are made, followed by a sale:
- Buy 10 NVDA at $100 → one lot
- Buy 5 NVDA at $110 → second lot
- Sell 12 NVDA at $130

When you sell 12 NVDA:
- You take 10 from the first lot (oldest).
- You take 2 from the second lot.
- The second lot now has 3 units remaining.

The realized gain/loss is computed **per used lot**, and each piece becomes a
row on Form 8949.

---

## Transaction blocks (`Tx Index` and `Type`)

Each **logical transaction** in the CSV is represented by one or more rows
sharing the same:

- `Tx Index` – integer ID of the block (e.g. `0`, `1`, `2`, …)
- `Type` – transaction type for the entire block:
  - `"Buy"`
  - `"Sell"`
  - `"Exchange"`
  - `"Transfer"`

The pipeline groups the input DataFrame by `Tx Index` and expects **all rows in
a group to have the same `Type`**. If a block contains mixed types, it is
rejected with an error.

Within a block you can have:

- **Non-fee rows** – the actual economic legs, e.g.:
  - Spending USD, receiving NVDA
  - Selling NVDA, receiving USD
  - Swapping NVDA for TSLA
- **Fee rows** – rows where `Asset` starts with `"fee"` (e.g. `feeUSD`,
  `feeNVDA`, `feeTSLA`). These represent transaction fees paid in some asset.

A typical **Buy** block might look like this:

| Tx Index | Date       | Asset   | Amount (asset) | Sell price ($) | Buy price ($) | Type |
|---------:|------------|---------|----------------|----------------|---------------|------|
| 0        | 2024-09-04 | USD     | -1250.0        | 1.0            | 1.0           | Buy  |
| 0        | 2024-09-04 | NVDA    | 10.0           | NaN            | 125.0         | Buy  |
| 0        | 2024-09-04 | feeUSD  | -10.0          | 1.0            | NaN           | Buy  |

This is interpreted as:

- You **spend** USD (including a USD fee),
- You **receive** NVDA,
- Fees are folded into the cost basis of the NVDA lot according to the rules
  in the parsing logic.

The function `run_fifo_pipeline` processes each transaction block in turn,
extracts the **buy**, **sell**, and **fee** sides, and then updates the per-asset
FIFO ledger accordingly.

---

## Parsing buy, sell, and fee data

For each transaction block, the function

```python
parse_row_data(block_type, rows)
```

returns three AssetData structures:

```python
AssetData:
    asset: str | None
    amount: float
    price: float
    total: float  # cost or proceeds
    tx_date: date
```

### Buy side
- For a `"Buy"` or `"Exchange"` block, the **buy side** identifies the row
where `"Amount (asset)"` is **positive** (after subtracting fee rows in the same asset).
- `total` is the **total cost** including fees (when fees should be folded
into the buy).

### Sell side
- For a `"Sell"` or `"Exchange"` block, the **sell side** identifies the row
where `"Amount (asset)"` is **negative** (after subtracting fee rows).
- `total` is the **total proceeds** after any adjustments from fees.

### Fee side
- Any rows whose `"Asset"` starts with "fee" are treated as **fee rows**.
- Depending on the asset and type of transaction, fees may:
    - Be **added to the cost** of the bought asset,
    - Be **subtracted from proceeds** of the sold asset, or
    - Be treated as a **separate sale** (e.g. for transfers, or when fees are
paid in an asset that is not directly the buy or sell asset).

---

## Updating the FIFO ledger

Once `buy_data`, `sell_data`, and `fee_data` have been extracted for a block,
they are applied to the per-asset FIFO queues via `update_fifo(...)`.

At a high level:

1. **Buy updates**

   - If `buy_data.asset` is not `None` and not `"USD"`:
     - If `buy_data.amount > 0`  
       → append a new lot to that asset’s FIFO queue:

       ```python
       fifo[buy_data.asset].append({
           "amount": buy_data.amount,
           "price": buy_data.price,
           "cost": buy_data.total,
           "tx_date": buy_data.tx_date,
       })
       ```

     - If `buy_data.amount < 0`  
       → this means previously calculated fees (in the same asset) exceed the
         nominal buy amount, so the **net effect is a sale** of that asset.  
         In that case, the code calls `reduce_fifo(...)` to consume existing
         lots and emit Form 8949 rows:

       ```python
       reduce_fifo(
           form8949,
           abs(buy_data.amount),
           buy_data.asset,
           fifo[buy_data.asset],
           buy_data.total,
           buy_data.tx_date,
       )
       ```

2. **Sell updates**

   - If `sell_data.asset` is not `None`, not `"USD"`, and `sell_data.amount < 0`:
     - This represents a real sale of a non-USD asset.  
       The code again calls `reduce_fifo(...)` to:
       - Consume lots from the left (oldest first),
       - Compute proportional cost and proceeds,
       - Write one or more Form 8949 rows via `record_sale(...)`.

       ```python
       reduce_fifo(
           form8949,
           abs(sell_data.amount),
           sell_data.asset,
           fifo[sell_data.asset],
           sell_data.total,
           sell_data.tx_date,
       )
       ```

3. **Fee updates**

   - If `fee_data.asset` is the same as `buy_data.asset` or `sell_data.asset`,
     those fees **must already have been folded** into `buy_data` or
     `sell_data` upstream. In that case, `update_fifo` raises an error to avoid
     double-counting:

     ```python
     if fee_data.asset == buy_data.asset or fee_data.asset == sell_data.asset:
         raise ValueError(
             f"Fee asset {fee_data.asset} should already be taken "
             f"into account in buy {buy_data.asset} or sell "
             f"{sell_data.asset} asset."
         )
     ```

   - Otherwise, if `fee_data.asset` is a non-USD asset and  
     `fee_data.amount != 0.0`, paying the fee in that asset is treated as a
     **taxable event** where the fee asset would be sold to obtain dollars, 
     that in turn are used to pay the fee. The fee amount is sold out of the 
     FIFO lots for that asset:

     ```python
     if fee_data.asset is not None and fee_data.asset != "USD" and fee_data.amount != 0.0:
         reduce_fifo(
             form8949,
             abs(fee_data.amount),
             fee_data.asset,
             fifo[fee_data.asset],
             fee_data.total,
             fee_data.tx_date,
         )
     ```
---

## From FIFO movements to Form 8949 rows

Every time a lot (or part of a lot) is used in a sale, the code calls
`record_sale(...)` to create a **Form 8949-style row**.

For each portion of a lot that is sold, `record_sale` receives:

- `asset`: Ticker or symbol (e.g. `"NVDA"`).
- `amount`: Quantity of the asset sold from that lot.
- `proceeds`: Dollar proceeds allocated to this portion of the sale.
- `cost_basis`: Dollar cost basis allocated from the FIFO lot.
- `acquisition_date`: Original lot acquisition date.
- `sale_date`: Date of the sale.

Using that information, it appends a dictionary like this to `form8949`:

```python
{
    "Description": "10.00000000 NVDA",
    "Date Acquired": "01/01/2024",
    "Date Sold": "09/04/2024",
    "Proceeds": "10000.00",
    "Cost Basis": "1000.00",
    "Gain or Loss": "9000.00",
}
```

Key points:
- **One sale can produce multiple rows** if it consumes more than one FIFO lot.
- `"Proceeds"` and `"Cost Basis"` are rounded to cents and stored as strings,
matching Form 8949 formatting.
- `"Gain or Loss"` is computed as `"Proceeds" - "Cost Basis"`:
    - Positive values are written as a plain number (e.g. `"9000.00"`),
    - Negative values are wrapped in parentheses (e.g. `"(50.20)"`),
following IRS conventions.

All such dictionaries collected in `form8949` are:
- Returned by `run_fifo_pipeline(df)` as `List[Dict[str, str]]`, and
- Written by `main(...)` to "form8949.csv" for import into tax software
or manual transcription to Form 8949.

---

## Putting it all together

Here’s the full flow, end to end:

1. **You provide input**  
   You start with a CSV of raw transactions (for example, `asset_tx.csv`) with columns like:
   - `Date`
   - `Tx Index`
   - `Asset`
   - `Amount (asset)`
   - `Sell price ($)`
   - `Buy price ($)`
   - `Type`

2. **The pipeline groups rows into blocks**  
   `run_fifo_pipeline(df)` groups rows by:
   - `Tx Index` → which rows belong to the same logical transaction  
   - `Type` → `"Buy"`, `"Sell"`, `"Exchange"`, or `"Transfer"`

3. **Each block is parsed into three conceptual pieces**  
   For every block, `parse_row_data(...)` extracts:
   - `buy_data` → what you acquired and at what cost  
   - `sell_data` → what you disposed of and for how much  
   - `fee_data` → any fees and which asset was used to pay them  

4. **FIFO ledgers are updated per asset**  
   `update_fifo(...)` uses those three pieces to mutate the per-asset FIFO queues:
   - **Buys** usually append new lots (amount, price, cost, date).
   - **Sells** consume existing lots from oldest to newest via `reduce_fifo(...)`.
   - **Fees** paid in non-USD assets can also trigger lot reductions (because paying fees in NVDA/TSLA/etc. is itself a taxable event).

5. **Each lot reduction generates Form 8949-style rows**  
   When a sale (or fee paid in-kind) consumes a lot, `record_sale(...)`:
   - Computes the proportional **cost basis** and **proceeds** for that slice.
   - Appends a dictionary with keys like:
     - `"Description"`
     - `"Date Acquired"`
     - `"Date Sold"`
     - `"Proceeds"`
     - `"Cost Basis"`
     - `"Gain or Loss"`

6. **The pipeline returns all rows as a list of dicts**  
   `run_fifo_pipeline(df)` returns a list of these Form 8949-style rows, which can be:
   - Written to CSV,
   - Further processed in Python,
   - Or inspected in tests.

7. **The CLI wrapper handles file IO**  
   The `main(...)` function is a thin IO layer:
   - Reads the input CSV into a DataFrame.
   - Calls `run_fifo_pipeline(df)` to compute gains and losses.
   - Writes the resulting rows to `form8949.csv`.

In short:

- **All tax logic lives in pure functions** like `run_fifo_pipeline`, `parse_row_data`, `update_fifo`, and `reduce_fifo`.  
- **All file handling** is kept in `main(...)`.  
