import pandas as pd
import pytest
from datetime import date, timedelta
from irs_asset_fifo_calculator import (calculate_taxes)
from collections import deque

# helpers
def data_is_equalish(data, expected):
    assert data.asset == expected.asset
    assert data.amount == pytest.approx(expected.amount, rel=1e-6)
    assert data.price == pytest.approx(expected.price, rel=1e-6)
    assert data.total == pytest.approx(expected.total, rel=1e-6)
    assert data.tx_date == expected.tx_date

def compare_parsed_rows(block_type: str, rows: pd.DataFrame, expected: tuple[AssetData, AssetData, AssetData]) -> None:
    buy_data, sell_data, fee_data = calculate_taxes.parse_row_data(block_type, rows)
    data_is_equalish(buy_data, expected[0])
    data_is_equalish(sell_data, expected[1])
    data_is_equalish(fee_data, expected[2])
    if block_type == 'transfer':
        assert buy_data.asset is None
        assert sell_data.asset is None

DEFAULT_TX_DATE = date(2024, 9, 4)

def AD(asset, amount, price, total, tx_date=DEFAULT_TX_DATE) -> AssetData:
    return AssetData(asset=asset, amount=amount, price=price, total=total, tx_date=tx_date)

# unit tests
@pytest.fixture(scope="function")
def form8949():
    return [{"Description": "10.00000000 NVDA",
            "Date Acquired": "1982-10-27",
            "Date Sold": "2024-12-31",
            "Proceeds": "10000",
            "Cost Basis": "1000",
            "Gain or Loss": "9000",
            "Code": "",
            "Adjustment Amount": ""}]

@pytest.fixture(scope="function")
def asset():
    return 'NVDA'

@pytest.fixture(scope="function")
def amount():
    return 10

@pytest.fixture(scope="function")
def proceeds():
    return 120

@pytest.fixture(scope="function")
def cost_basis():
    return 100

@pytest.fixture(scope="function")
def acquisition_date():
    return date(2024, 1, 1)

@pytest.fixture(scope="function")
def sale_date():
    return date(2024, 12, 31)

class TestRecordSale:
    def test_record_sale_success(self, form8949, asset, amount, proceeds,
            cost_basis, acquisition_date, sale_date):
        calculate_taxes.record_sale(form8949, asset, amount, proceeds,
                                    cost_basis, acquisition_date, sale_date)
        assert len(form8949) == 2
        assert  form8949[1]["Description"] == "10.00000000 " + asset
        assert  form8949[1]["Date Acquired"] == "2024-01-01"
        assert  form8949[1]["Date Sold"] == "2024-12-31"
        assert  form8949[1]["Proceeds"] == "120.00"
        assert  form8949[1]["Cost Basis"] == "100.00"
        assert  form8949[1]["Gain or Loss"] == "20.00"
        assert  form8949[1]["Code"] == ""
        assert  form8949[1]["Adjustment Amount"] == ""

    def test_record_sale_small_proceeds_and_cost_basis(self, form8949, asset,
                amount, acquisition_date, sale_date):
        proceeds = 0.0049
        cost_basis = 0.0049
        calculate_taxes.record_sale(form8949, asset, amount, proceeds,
                                    cost_basis, acquisition_date, sale_date)
        assert len(form8949) == 1

        proceeds = 0.005
        cost_basis = 0.0049
        calculate_taxes.record_sale(form8949, asset, amount, proceeds,
                                    cost_basis, acquisition_date, sale_date)
        assert len(form8949) == 2

    def test_record_sale_equal_dates(self, form8949, asset, amount, proceeds,
                cost_basis, acquisition_date):
        sale_date = acquisition_date
        calculate_taxes.record_sale(form8949, asset, amount, proceeds,
                                    cost_basis, acquisition_date, sale_date)
        assert len(form8949) == 2
        assert  form8949[1]["Date Acquired"] == form8949[1]["Date Sold"]

    def test_record_sale_loss(self, form8949, asset, amount, proceeds,
            acquisition_date, sale_date):
        cost_basis = proceeds + 100
        calculate_taxes.record_sale(form8949, asset, amount, proceeds,
                                    cost_basis, acquisition_date, sale_date)
        assert len(form8949) == 2
        assert form8949[1]["Gain or Loss"] == "-100.00"

    def test_record_sale_rounding(self, form8949, asset, amount, proceeds,
                cost_basis, acquisition_date, sale_date):
        proceeds = 120.9999
        cost_basis = 100.001
        calculate_taxes.record_sale(form8949, asset, amount, proceeds,
                                    cost_basis, acquisition_date, sale_date)
        assert len(form8949) == 2
        assert form8949[1]["Proceeds"] == "121.00"
        assert form8949[1]["Cost Basis"] == "100.00"
        assert len(form8949[1]["Gain or Loss"].rsplit('.')[-1]) == 2

    def test_record_sale_none_return(self, form8949, asset, amount, proceeds,
                cost_basis, acquisition_date, sale_date):
        assert calculate_taxes.record_sale(form8949, asset, amount,
                                           proceeds, cost_basis, acquisition_date, sale_date) is None
        assert len(form8949) == 2

    def test_record_sale_negative_amount(self, form8949, asset,
                proceeds, cost_basis, acquisition_date, sale_date, readout):
        amount = -1
        calculate_taxes.record_sale(form8949, asset, amount,
                                    proceeds, cost_basis, acquisition_date, sale_date)
        assert len(form8949) == 2
        assert  form8949[1]["Description"] == "1.00000000 " + asset

        out = readout()
        assert "Amount must be greater than zero." in out
        assert "is set as absolute." in out

    def test_record_sale_negative_proceeds(self, form8949, asset, amount,
                cost_basis, acquisition_date, sale_date, readout):
        proceeds = -1
        calculate_taxes.record_sale(form8949, asset, amount,
                                    proceeds, cost_basis, acquisition_date, sale_date)
        assert len(form8949) == 2
        assert  form8949[1]["Proceeds"] == "1.00"

        out = readout()
        assert "Proceeds must be greater than zero." in out
        assert "is set as absolute." in out

    def test_record_sale_negative_cost_basis(self, form8949, asset, amount,
                proceeds, cost_basis, acquisition_date, sale_date, readout):
        cost_basis = -1
        calculate_taxes.record_sale(form8949, asset, amount,
                                    proceeds, cost_basis, acquisition_date, sale_date)
        assert len(form8949) == 2
        assert  form8949[1]["Cost Basis"] == "1.00"

        out = readout()
        assert "Cost basis must be greater than zero." in out
        assert "is set as absolute." in out

    def test_record_sale_date_order(self, form8949, asset, amount,
                proceeds, cost_basis, sale_date):
        acquisition_date = sale_date + timedelta(days=1)

        with pytest.raises(ValueError, match="Acquisition date must be "
                + "before sale date."):
            calculate_taxes.record_sale(form8949, asset, amount,
                                        proceeds, cost_basis, acquisition_date, sale_date)

    def test_record_sale_non_datetime_acquisition(self, form8949, asset,
                amount, proceeds, cost_basis, sale_date):
        acquisition_date = "2024/31/10"

        with pytest.raises(TypeError, match="Acquisition date must be "
                                            + "in date format."):
            calculate_taxes.record_sale(form8949, asset, amount,
                                        proceeds, cost_basis, acquisition_date, sale_date)

    def test_record_sale_non_datetime_sale(self, form8949, asset,
                amount, proceeds, cost_basis, acquisition_date):
        sale_date = "2024/31/10"

        with pytest.raises(TypeError, match="Sale date must be "
                                            + "in date format."):
            calculate_taxes.record_sale(form8949, asset, amount,
                                        proceeds, cost_basis, acquisition_date, sale_date)

    def test_record_sale_non_float_amount(self, form8949, asset,
                proceeds, cost_basis, acquisition_date, sale_date):
        amount = "five"

        with pytest.raises(TypeError, match=r"Amounts \(\$ and asset\) must "
                + "be in float format\.\s+.* sale on .* is invalid\."):
            calculate_taxes.record_sale(form8949, asset, amount,
                                        proceeds, cost_basis, acquisition_date, sale_date)

    def test_record_sale_non_float_proceeds(self, form8949, asset,
                amount, cost_basis, acquisition_date, sale_date):
        proceeds = "five"

        with pytest.raises(TypeError, match=r"Amounts \(\$ and asset\) must "
                + "be in float format\.\s+.* sale on .* is invalid\."):
            calculate_taxes.record_sale(form8949, asset, amount,
                                        proceeds, cost_basis, acquisition_date, sale_date)

    def test_record_sale_non_float_cost_basis(self, form8949, asset,
                amount, proceeds, acquisition_date, sale_date):
        cost_basis = "five"

        with pytest.raises(TypeError, match=r"Amounts \(\$ and asset\) must "
                + "be in float format\.\s+.* sale on .* is invalid\."):
            calculate_taxes.record_sale(form8949, asset, amount,
                                        proceeds, cost_basis, acquisition_date, sale_date)

    def test_record_sale_non_list_form(self, asset, amount,
                proceeds, cost_basis, acquisition_date, sale_date):
        form8949 = dict()

        with pytest.raises(TypeError, match="A list object must be passed. "
                + "Create form8949 list first."):
            calculate_taxes.record_sale(form8949, asset, amount,
                                        proceeds, cost_basis, acquisition_date, sale_date)


@pytest.fixture(scope="function")
def fifo():
    return {'NVDA': deque([{"amount": 10, "price": 100, "cost": 1000,
                            "tx_date": date(2024, 1, 1)},
                           {"amount": 5, "price": 110,
                            "cost": (5 * 110) * 1.002,
                            "tx_date": date(2024, 2, 1)},
                           {"amount": 2, "price": 80,
                            "cost": (2 * 80) * 1.002,
                            "tx_date": date(2024, 3, 1)}])
            }

class TestUpdateFifo:

    @pytest.mark.parametrize(
        "sell_amount, expected",
        [
            (10, {'length': 2, 'amount': 5, 'price': 110, 'cost': (5 * 110) * 1.002,
                  "tx_date": date(2024, 2, 1)}),
            (9, {'length': 3, 'amount': 10 - 9, 'price': 100, 'cost': 1000 * (10 - 9)/10,
                 "tx_date": date(2024, 1, 1)}),
            (15, {'length': 1, 'amount': 10 + 5 - 15 + 2, 'price': 80, 'cost': (2 * 80) * 1.002,
                  "tx_date": date(2024, 3, 1)}),
            (0, {'length': 3, 'amount': 10, 'price': 100, 'cost': 1000,
                 "tx_date": date(2024, 1, 1)}),
            (-1, {'length': 3, 'amount': 10 - 1, 'price': 100, 'cost': 1000 * (10 - 1) / 10,
                  "tx_date": date(2024, 1, 1)}),
            (0.00000001, {'length': 3, 'amount': 10 - 0.00000001, 'price': 100, 'cost': 1000 * (10 - 0.00000001) / 10,
                          "tx_date": date(2024, 1, 1)})
        ],
        ids=["sell-10", "sell-9", "sell-15", "sell-0", "sell-neg1", "sell-tiny"]
    )

    def test_update_fifo_sell_amount(self, sell_amount, expected, asset, fifo):
        form8949 = list()
        sell_price = 150
        calculate_taxes.update_fifo(form8949, sell_amount, asset, fifo, sell_amount*sell_price,
                                    date(2024, 4, 1))
        assert len(fifo[asset]) == expected['length']
        assert fifo[asset][0]['amount'] == pytest.approx(expected['amount'], 0, 1e-6)
        assert fifo[asset][0]['price'] == pytest.approx(expected['price'], 0, 1e-6)
        assert fifo[asset][0]['cost'] == pytest.approx(expected['cost'], 0, 1e-6)
        assert fifo[asset][0]['tx_date'] == expected['tx_date']

        if expected['length'] == 3:
            # check that 2nd and 3rd lost remain unchanged
            assert fifo[asset][1]['amount'] == pytest.approx(5, 1e-6, 0)
            assert fifo[asset][1]['price'] == pytest.approx(110, 1e-6, 0)
            assert fifo[asset][1]['cost'] == pytest.approx((5 * 110) * 1.002, 1e-6, 0)
            assert fifo[asset][1]['tx_date'] == date(2024, 2, 1)
            assert fifo[asset][2]['amount'] == pytest.approx(2, 1e-6, 0)
            assert fifo[asset][2]['price'] == pytest.approx(80, 1e-6, 0)
            assert fifo[asset][2]['cost'] == pytest.approx((2 * 80) * 1.002, 1e-6, 0)
            assert fifo[asset][2]['tx_date'] == date(2024, 3, 1)

            # check that form8949 is written correctly
            if sell_amount == 9 and expected == {'length': 3, 'amount': 10 - 9, 'price': 100, 'cost': 1000 * (10 - 9)/10,
                 "tx_date": date(2024, 1, 1)}:
                assert form8949[0]['Description'] == f"{round(sell_amount, 8):.8f}" + " " + asset
                assert form8949[0]['Date Acquired'] == f"2024-01-01"
                assert form8949[0]['Date Sold'] == f"2024-04-01"
                assert form8949[0]['Proceeds'] == f"{round(1350, 2):.2f}"
                assert form8949[0]['Cost Basis'] == f"{round(900, 2):.2f}"
                assert form8949[0]['Gain or Loss'] == f"{round(450, 2):.2f}"
                assert form8949[0]['Code'] == ""
                assert form8949[0]['Adjustment Amount'] == ""

    def test_update_fifo_missing_key(self, asset, amount, proceeds,
            sale_date, fifo):
        del fifo[asset][0]['amount']
        with pytest.raises(KeyError, match=r"contains an invalid"
                           + " purchase."):
            calculate_taxes.update_fifo([], amount, asset, fifo, proceeds,
                                        sale_date)

    def test_update_fifo_type_error(self, asset, amount, proceeds,
            sale_date, fifo):
        fifo[asset][0]['amount'] = 'five'
        with pytest.raises(TypeError, match=r"contains an invalid"
                           + " purchase."):
            calculate_taxes.update_fifo([], amount, asset, fifo, proceeds,
                                        sale_date)

    def test_update_fifo_small_lot_amount(self, asset, fifo):
        """
        We reduce the tiny first lot to zero, then continue selling
        # from the second lot; the remaining amount after selling 4
        from the 5-unit second lot + tiny first lot is 1.00001
        """
        fifo[asset][0]['amount'] = 0.00001
        calculate_taxes.update_fifo([], 4, asset, fifo, 100,
                                    date(2024, 4, 1))
        assert len(fifo[asset]) == 2
        assert fifo[asset][0]['amount'] == pytest.approx(1.00001, rel=0, abs=1e-8)
        assert fifo[asset][0]['price'] == pytest.approx(110, rel=0, abs=1e-6)
        assert fifo[asset][0]['cost'] == pytest.approx(550*1.002*1.00001/5, rel=0, abs=1e-6)
        assert fifo[asset][0]['tx_date'] == date(2024, 2, 1)

    def test_update_fifo_missing_asset(self, asset, amount, proceeds,
            sale_date, fifo):
        del fifo[asset]
        with pytest.raises(ValueError, match=f"does not contain"):
            calculate_taxes.update_fifo([], amount, asset, fifo, proceeds,
                                        sale_date)

@pytest.fixture(scope="function")
def row0():
    return pd.Series({'Date': '5 / 22 / 2025', 'Asset': 'USD',
                      'Amount (asset)': -1250.0, 'Sell price ($)': 1.0,
                      'Buy price ($)': 1.0, 'Account number': 1234,
                      'Entity': 'Chase', 'Notes': '', 'Remaining': '',
                      'Tx Date': '2024-09-04'})

@pytest.fixture(scope="function")
def row1():
    return pd.Series({'Date': '5 / 22 / 2025', 'Asset': 'NVDA',
                      'Amount (asset)': 10.0, 'Sell price ($)': 'NaN',
                      'Buy price ($)': 12.0, 'Account number': 1234,
                      'Entity': 'Chase', 'Notes': '', 'Remaining': '',
                      'Tx Date': '2024-09-04'})

class TestDefineAmounts:

    @pytest.mark.parametrize(
        "series_amounts",
        [
            ('-1250', '10.0'),
            ('-1,250', '10.0'),
            ('-1250,', '10.0'),
            (' - 1,250.00, ', '10.0'),
            (' -1,250.00, ', ' 10.0, ')
        ],
        ids=["normal_amount", "comma_separator_amount", "trailing_comma_amount",
             "padded_comma_separator_decimals_trailing_comma", "row1_amount"]
    )

    def test_define_amounts_success(self, series_amounts, row0, row1):
        row0['Amount (asset)'] = series_amounts[0]
        row1['Amount (asset)'] = series_amounts[1]
        amount0, amount1 = calculate_taxes.define_amounts(row0, row1)
        assert amount0 == pytest.approx(-1250, rel=1e-6, abs=0)
        assert amount1 == pytest.approx(10, rel=1e-6, abs=0)

    def test_define_amounts_non_number_row0(self, row0, row1):
        row0['Amount (asset)'] = 'blah'
        with pytest.raises(ValueError, match="Invalid amount"):
            calculate_taxes.define_amounts(row0, row1)

    def test_define_amounts_non_number_row1(self, row0, row1):
        row1['Amount (asset)'] = 'blah'
        with pytest.raises(ValueError, match="Invalid amount"):
            calculate_taxes.define_amounts(row0, row1)

    def test_define_amounts_wrong_type(self, row0, row1):
        row0['Amount (asset)'] = []
        with pytest.raises(TypeError, match="Invalid amount"):
            calculate_taxes.define_amounts(row0, row1)


class TestDefineBlocks:
    @pytest.mark.parametrize(
        "account_number0, asset0, asset1, amnt0, amnt1, expected",
        [
            ('Approved', 'feeNVDA', 'USD', -1.0, -100.0, ('approved_exchange', 4)),
            ('1234-5688', 'NVDA', 'feeUSD', 50.0, -2.0, ('transfer', 2)),
            ('1234', 'USD', 'NVDA', -100.0, 8.0, ('purchase', 3)),
            ('1234', 'NVDA', 'USD', -8.0, 100.0, ('sale', 3)),
            ('1234', 'TSLA', 'NVDA', -100.0, 8.0, ('exchange', 3)),
            ("1234", "USD", "NVDA", "-1,250.0", "10.0", ("purchase", 3)),
            ('Approved', 'TSLA', 'NVDA', -100.0, 8.0, ('exchange', 3)),
            ('1234-5688', 'USD', 'feeUSD', 50.0, -2.0, ('transfer', 2)),
        ],
        ids=['approved_exchange', 'transfer', 'purchase', 'sale', 'exchange',
             'dirty_purchase_amount', 'approved_no_fee', 'usd_transfer']
    )

    def test_define_blocks_success(self, account_number0, asset0, asset1, amnt0,
                                   amnt1, expected, row0, row1):
        row0['Account number'] = account_number0
        row0['Asset'] = asset0
        row1['Asset'] = asset1
        row0['Amount (asset)'] = amnt0
        row1['Amount (asset)'] = amnt1
        block_type, n_tx = calculate_taxes.define_blocks(row0, row1)
        assert block_type == expected[0]
        assert n_tx == expected[1]

    @pytest.mark.parametrize(
        "asset0, asset1, amnt0, amnt1",
        [
            ('NVDA', 'TSLA', 5, 10.0),
            ('NVDA', 'TSLA', 5, -10.0),
            ('USD', 'NVDA', -100.0, -8.0),
            ('NVDA', 'USD', 100.0, 8.0),
        ],
        ids=['two_positive_amounts', 'inverted_exchange_amounts',
             'negative_purchase', 'positive_sale']
    )

    def test_define_blocks_fail(self, asset0, asset1, amnt0, amnt1, row0, row1):
        row0['Asset'] = asset0
        row1['Asset'] = asset1
        row0['Amount (asset)'] = amnt0
        row1['Amount (asset)'] = amnt1
        with pytest.raises(ValueError, match="Invalid block: could not "
                           + "classify transaction pair"):
            calculate_taxes.define_blocks(row0, row1)

@pytest.fixture(scope='function')
def rows():
    return pd.DataFrame({'Date': ['9/4/2024'] * 4,
                         'Asset': ['feeUSD', 'NVDA', 'USD', 'feeUSD'],
                         'Amount (asset)': [-5, -10, 1250, -5],
                         'Sell price ($)': [1, 125, float("nan"), 1],
                         'Buy price ($)': [float("nan"), float("nan"), 1, float("nan")],
                         'Account number': [1234] * 4,
                         'Entity': ['Chase'] * 4,
                         'Notes': [''] * 4,
                         'Remaining': [''] * 4,
                         'Tx Date': [date(2024,9,4)] * 4})

class TestCheckFees:
    @pytest.mark.parametrize(
        "block_type, first_asset, mid_asset, last_asset, expected_match",
        [
            ('approved_exchange', 'feeNVDA', 'USD', 'feeUSD', None),
            ('purchase', 'NVDA', 'USD', 'feeUSD', None),
            ('transfer', 'NVDA', 'USD', 'feeUSD', None),
            ("sale", "NVDA", "USD", "feeUSD", None),
            ('purchase', 'feeNVDA', 'USD', 'feeUSD', 'Invalid block: extra fee'),
            ('approved_exchange', 'NVDA', 'USD', 'feeUSD', 'Invalid block: missing approval fee'),
            ("sale", "NVDA", "USD", "feeUS", "Invalid block: missing fee"),  # too short to count as fee
            ('sale', 'NVDA', 'USD', 'FEED', 'Invalid block: missing fee'),
            ('sale', 'NVDA', 'feeUSD', 'feeNVDA', 'Invalid block: extra fee'),
        ],
        ids=['approved_success', 'purchase_success', 'transfer_success', 'sale_success',
             'purchase_first_is_fee', 'approved_no_first_fee', 'sale_short_fee', 'sale_FEED',
             'sale_mid_fee']
    )

    def test_check_fees(self, rows, block_type, first_asset, mid_asset, last_asset, expected_match):
        if block_type == 'approved_exchange':
            n_rows = 4
        elif block_type == 'transfer':
            n_rows = 2
        else:
            n_rows = 3
        this_rows = rows.head(n_rows).copy()
        this_rows.loc[0, 'Asset'] = first_asset
        if n_rows > 2:
            this_rows.loc[1, 'Asset'] = mid_asset
        if n_rows > 3:
            this_rows.loc[2, 'Asset'] = mid_asset
        this_rows.loc[this_rows.index[-1], 'Asset'] = last_asset
        if not expected_match:
            assert calculate_taxes.check_fees(block_type, this_rows) is None
        else:
            with pytest.raises(ValueError, match=expected_match):
                calculate_taxes.check_fees(block_type, this_rows)

    def test_check_fees_empty_df(self):
        this_rows = pd.DataFrame()
        with pytest.raises(ValueError, match="Empty dataframe."):
            calculate_taxes.check_fees('purchase', this_rows)

class TestParseRowData:

    # check approved_exchange path
    @pytest.mark.parametrize(
    "fee_asset1, fee_asset2, amount_fee_asset1, amount_fee_asset2, price_fee_asset1, price_fee_asset2, expected, expected_error, expected_error_message", [
            ('feeUSD', 'feeUSD', -6.0, -4.0, 1.0, 1.0,
             (AD('USD', 1240.0, 1.0, 1260.0),
              AD('NVDA', -10.0, 125.0, 1240.0),
              AD('USD', -10.0, 1.0,  10.0)),
             None, ""
            ),
            ('feeTSLA', 'feeTSLA', -0.2, -0.3, 50.0, 50.0,
             (AD('USD', 1250.0, 1.0, 1275.0),
              AD('NVDA', -10.0, 125.0, 1225.0),
              AD('TSLA', -0.5, 50.0, 25.0)),
             None, ""
            ),
            ('feeUSD', 'feeNVDA', -6.0, -0.32, 1.0, 125.0, None,
             ValueError, "Fee asset mismatch."
             ),
            ('feeTSLA', 'feeUSD', -0.3, -4.0, 50.0, 1.0, None,
             ValueError, "Fee asset mismatch."
             ),
            ('feeTSLA', 'feeEUR', -6.0, -4.0, 50.0, 1.15, None,
             ValueError, "Fee asset mismatch."
             ),
            ('USD', 'USD', -6.0, -4.0, 1.0, 1.0, None,
             ValueError, "is not a valid fee element"
             ),

        ],
        ids=['approved_same_fee_asset_among_trading_pair', 'approved_same_fee_asset_different_from_trading_pair',
             'approved_different_fee_assets_same_as_trading_pair', 'approved_different_fee_assets_one_among_trading_pair',
             'approved_different_fee_assets_different_from_trading_pair', 'approved_same_invalid_fee_asset']
    )

    def test_parse_row_data_approved(self, fee_asset1, fee_asset2, amount_fee_asset1, amount_fee_asset2, price_fee_asset1, price_fee_asset2, expected, expected_error,
                                    expected_error_message, rows):
        block_type = 'approved_exchange'
        rows.loc[0, 'Asset'] = fee_asset1
        rows.loc[3, 'Asset'] = fee_asset2
        rows.loc[0, 'Amount (asset)'] = amount_fee_asset1
        rows.loc[3, 'Amount (asset)'] = amount_fee_asset2
        rows.loc[0, 'Sell price ($)'] = price_fee_asset1
        rows.loc[3, 'Sell price ($)'] = price_fee_asset2

        if expected_error is not None:
            with pytest.raises(expected_error, match=expected_error_message):
                calculate_taxes.parse_row_data(block_type, rows)
        else:
            compare_parsed_rows(block_type, rows, expected)

    # check exchange path
    @pytest.mark.parametrize(
    "fee_asset, amount_fee_asset, price_fee_asset, buy_asset, buy_amount, buy_price, expected", [
            ('feeUSD', -10.0, 1.0, 'TSLA', 25.0, 50.0, (
              AD('TSLA', 25.0, 50.0, 1260.0),
              AD('NVDA', -10.0, 125.0, 1240.0),
              AD('USD', -10.0, 1.0, 10.0)),
            ),
            ('feeTSLA', -0.4, 50.0, 'TSLA', 25.0, 50.0,
             (AD('TSLA', 24.6, 50.0, 1270.0),
              AD('NVDA', -10.0, 125.0, 1230.0),
              AD('TSLA', -0.4, 50.0, 20.0)),
             ),
            ('feeNVDA', -0.1, 125.0, 'TSLA', 25.0, 50.0,
             (AD('TSLA', 25.0, 50.0, 1262.5),
              AD('NVDA', -10.1, 125.0, 1237.5),
              AD('NVDA', -0.1, 125.0, 12.5)),
             ),
            ('feeTSLA', -26, 50.0, 'TSLA', 25.0, 50.0,
             (AD('TSLA', -1.0, 50.0, 2550.0),
              AD('NVDA', -10.0, 125.0, -50.0),
              AD('TSLA', -26.0, 50.0, 1300.0)),
             ),
        ],
        ids=['exchange_different_fee_asset', 'exchange_same_fee_asset_as_buy',
             'exchange_same_fee_asset_as_sale', 'exchange_fee_exceeds_buy'
             ]
    )

    def test_parse_row_data_exchange(self, fee_asset, amount_fee_asset, price_fee_asset, buy_asset, buy_amount, buy_price, expected,
                                    rows):
        block_type = 'exchange'
        rows = rows.drop(0)
        rows = rows.reset_index(drop=True)
        rows.loc[1, 'Asset'] = buy_asset
        rows.loc[2, 'Asset'] = fee_asset
        rows.loc[1, 'Amount (asset)'] = buy_amount
        rows.loc[2, 'Amount (asset)'] = amount_fee_asset
        rows.loc[1, 'Buy price ($)'] = buy_price
        rows.loc[2, 'Sell price ($)'] = price_fee_asset

        compare_parsed_rows(block_type, rows, expected)

    # check purchase path
    @pytest.mark.parametrize(
    "fee_asset, amount_fee_asset, price_fee_asset, buy_asset, buy_amount, buy_price, expected", [
            ('feeUSD', -10.0, 1.0, 'TSLA', 25.0, 50.0,
             (AD('TSLA', 25.0, 50.0, 1260.0),
              AD('USD', -1260.0, 1.0, 0.0),
              AD('USD', -10.0, 1.0, 10.0)),
            ),
            ('feeUSD', -1250.0, 1.0, 'TSLA', 25.0, 50.0,
             (AD('TSLA', 25.0, 50.0, 2500.0),
              AD('USD', -2500.0, 1.0, 0.0),
              AD('USD', -1250.0, 1.0, 1250.0)),
            ),
            ('feeUSD', -1260.0, 1.0, 'TSLA', 25.0, 50.0,
             (AD('TSLA', 25.0, 50.0, 2510.0),
              AD('USD', -2510.0, 1.0, 0.0),
              AD('USD', -1260.0, 1.0, 1260.0)
              ),
            ),
            ('feeTSLA', -1.0, 50.0, 'TSLA', 25.0, 50.0,
             (AD('TSLA', 24.0, 50.0, 1300.0),
              AD('USD', -1250.0, 1.0, 0.0),
              AD('TSLA', -1.0, 50.0, 50.0)
              ),
             ),
            ('feeTSLA', -25.0, 50.0, 'TSLA', 25.0, 50.0,
             (AD('TSLA', 0.0, 50.0, 2500.0),
              AD('USD', -1250.0, 1.0, 0.0),
              AD('TSLA', -25.0, 50.0, 1250.0)
              ),
             ),
            ('feeTSLA', -26.0, 50.0, 'TSLA', 25.0, 50.0,
             (AD('TSLA', -1.0, 50.0, 2550.0),
              AD('USD', -1250.0, 1.0, 0.0),
              AD('TSLA', -26.0, 50.0, 1300.0)
              ),
             ),
            ('feeNVDA', -0.2, 125.0, 'TSLA', 25.0, 50.0,
             (AD('TSLA', 25.0, 50.0, 1275.0),
              AD('USD', -1250.0, 1.0, 0.0),
              AD('NVDA', -0.2, 125.0, 25.0)),
             ),
        ],
        ids=['purchase_fee_asset_same_as_sell', 'purchase_fee_same_as_sell', 'purchase_fee_exceeds_sell',
             'purchase_fee_asset_same_as_buy', 'purchase_fee_same_as_buy', 'purchase_fee_exceeds_buy',
             'purchase_different_fee_asset'
             ]
    )

    def test_parse_row_data_purchase(self, fee_asset, amount_fee_asset, price_fee_asset, buy_asset, buy_amount, buy_price, expected,
                                     rows):
        block_type = 'purchase'
        rows = rows.drop(0)
        rows = rows.reset_index(drop=True)
        rows.loc[0, 'Asset'] = 'USD'
        rows.loc[1, 'Asset'] = buy_asset
        rows.loc[2, 'Asset'] = fee_asset
        rows.loc[0, 'Amount (asset)'] = -buy_amount*buy_price
        rows.loc[1, 'Amount (asset)'] = buy_amount
        rows.loc[2, 'Amount (asset)'] = amount_fee_asset
        rows.loc[0, 'Sell price ($)'] = 1.0
        rows.loc[1, 'Buy price ($)'] = buy_price
        rows.loc[2, 'Sell price ($)'] = price_fee_asset

        compare_parsed_rows(block_type, rows, expected)

    # check sale path
    @pytest.mark.parametrize(
    "fee_asset, amount_fee_asset, price_fee_asset, expected", [
            ('feeUSD', -10.0, 1.0,
             (AD('USD', 1240.0, 1.0, 0.0),
              AD('NVDA', -10.0, 125.0, 1240.0),
              AD('USD', -10.0, 1.0, 10.0)),
            ),
            ('feeUSD', -1260.0, 1.0,
             (AD('USD', -10.0, 1.0, 0.0),
              AD('NVDA', -10.0, 125.0, -10.0),
              AD('USD', -1260.0, 1.0, 1260.0)),
            ),
            ('feeNVDA', -0.2, 125.0,
             (AD('USD', 1250.0, 1.0, 0.0),
              AD('NVDA', -10.2, 125.0, 1225.0),
              AD('NVDA', -0.2, 125.0, 25.0)
              ),
             ),
            ('feeNVDA', -11.0, 125.0,
             (AD('USD', 1250.0, 1.0, 0.0),
              AD('NVDA', -21.0, 125.0, -125.0),
              AD('NVDA', -11.0, 125.0, 1375.0)
              ),
             ),
        ],
        ids=['sale_same_fee_asset_as_buy', 'sale_fee_exceeds_buy', 'sale_same_fee_asset_as_sale', 'sale_fee_exceeds_sale'
             ]
    )

    def test_parse_row_data_sale(self, fee_asset, amount_fee_asset, price_fee_asset, expected, rows):
        block_type = 'sale'
        rows = rows.drop(0)
        rows = rows.reset_index(drop=True)
        rows.loc[2, 'Asset'] = fee_asset
        rows.loc[2, 'Amount (asset)'] = amount_fee_asset
        rows.loc[2, 'Sell price ($)'] = price_fee_asset

        compare_parsed_rows(block_type, rows, expected)

    # check transfer path
    @pytest.mark.parametrize(
    "fee_asset, amount_fee_asset, price_fee_asset, expected", [
            ('feeTSLA', -0.1, 50.0,
             (AD(None, 0.0, 0.0, 0.0),
              AD(None, 0.0, 0.0, 0.0),
              AD('TSLA', -0.1, 50.0, 5.0)
              ),
            ),
            ('feeUSD', -10, 1.0,
             (AD(None, 0.0, 0.0, 0.0),
              AD(None, 0.0, 0.0, 0.0),
              AD('USD', -10.0, 1.0, 10.0)
              ),
            ),
            ('feeUSD', -1260, 1.0,
             (AD(None, 0.0, 0.0, 0.0),
              AD(None, 0.0, 0.0, 0.0),
              AD('USD', -1260.0, 1.0, 1260.0)
              ),
            ),
        ],
        ids=['transfer_different_fee_asset', 'transfer_same_assets', 'transfer_fee_exceeds_buy'
             ]
    )

    def test_parse_row_data_transfer(self, fee_asset, amount_fee_asset, price_fee_asset, expected, rows):
        block_type = 'transfer'
        rows = rows.drop(0)
        rows = rows.reset_index(drop=True)
        rows = rows.drop(0)
        rows = rows.reset_index(drop=True)
        rows.loc[1, 'Asset'] = fee_asset
        rows.loc[1, 'Amount (asset)'] = amount_fee_asset
        rows.loc[1, 'Sell price ($)'] = price_fee_asset

        compare_parsed_rows(block_type, rows, expected)
