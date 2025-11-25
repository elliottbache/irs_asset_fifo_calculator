import pandas as pd
import pytest
from datetime import datetime, timedelta
from irs_asset_fifo_calculator import (calculate_taxes)
from collections import deque, defaultdict


# helpers
"""
class FakeCompleted:
    def __init__(self, stdout="", stderr="", returncode=0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode
"""


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
    return datetime(2024, 1, 1)

@pytest.fixture(scope="function")
def sale_date():
    return datetime(2024, 12, 31)

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
                                            + "in datetime format."):
            calculate_taxes.record_sale(form8949, asset, amount,
                                        proceeds, cost_basis, acquisition_date, sale_date)

    def test_record_sale_non_datetime_sale(self, form8949, asset,
                amount, proceeds, cost_basis, acquisition_date):
        sale_date = "2024/31/10"

        with pytest.raises(TypeError, match="Sale date must be "
                                            + "in datetime format."):
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
    return {'NVDA': deque([{"amount": 10, "price": 10, "cost": 100,
                            "timestamp": datetime(2024, 1, 1)},
                           {"amount": 5, "price": 11,
                            "cost": (5 * 11) * 1.002,
                            "timestamp": datetime(2024, 2, 1)},
                           {"amount": 2, "price": 8,
                            "cost": (2 * 8) * 1.002,
                            "timestamp": datetime(2024, 3, 1)}])
            }

class TestUpdateFifo:

    @pytest.mark.parametrize(
        "sell_amount, expected",
        [
            (10, {'length': 2, 'amount': 5, 'price': 11, 'cost': (5 * 11) * 1.002,
                  "timestamp": datetime(2024, 2, 1)}),
            (9, {'length': 3, 'amount': 10 - 9, 'price': 10, 'cost': 100 * (10 - 9) / 10,
                 "timestamp": datetime(2024, 1, 1)}),
            (15, {'length': 1, 'amount': 10 + 5 - 15 + 2, 'price': 8, 'cost': (2 * 8) * 1.002,
                  "timestamp": datetime(2024, 3, 1)}),
            (0, {'length': 3, 'amount': 10, 'price': 10, 'cost': 100,
                 "timestamp": datetime(2024, 1, 1)}),
            (-1, {'length': 3, 'amount': 10 - 1, 'price': 10, 'cost': 100 * (10 - 1) / 10,
                  "timestamp": datetime(2024, 1, 1)}),
            (0.00000001, {'length': 3, 'amount': 10 - 0.00000001, 'price': 10, 'cost': 100 * (10 - 0.00000001) / 10,
                          "timestamp": datetime(2024, 1, 1)})
        ],
        ids=["sell-10", "sell-9", "sell-15", "sell-0", "sell-neg1", "sell-tiny"]
    )

    def test_update_fifo_sell_amount(self, sell_amount, expected, asset, fifo):
        form8949 = list()
        calculate_taxes.update_fifo(form8949, sell_amount, asset, fifo, 100,
                                    datetime(2024, 4, 1))
        assert len(fifo[asset]) == expected['length']
        assert fifo[asset][0]['amount'] == pytest.approx(expected['amount'], 0, 1e-6)
        assert fifo[asset][0]['price'] == pytest.approx(expected['price'], 0, 1e-6)
        assert fifo[asset][0]['cost'] == pytest.approx(expected['cost'], 0, 1e-6)
        assert fifo[asset][0]['timestamp'] == expected['timestamp']

        if expected['length'] == 3:
            # check that 2nd and 3rd lost remain unchanged
            assert fifo[asset][1]['amount'] == pytest.approx(5, 1e-6, 0)
            assert fifo[asset][1]['price'] == pytest.approx(11, 1e-6, 0)
            assert fifo[asset][1]['cost'] == pytest.approx((5 * 11) * 1.002, 1e-6, 0)
            assert fifo[asset][1]['timestamp'] == datetime(2024, 2, 1)
            assert fifo[asset][2]['amount'] == pytest.approx(2, 1e-6, 0)
            assert fifo[asset][2]['price'] == pytest.approx(8, 1e-6, 0)
            assert fifo[asset][2]['cost'] == pytest.approx((2 * 8) * 1.002, 1e-6, 0)
            assert fifo[asset][2]['timestamp'] == datetime(2024, 3, 1)

            # check that form8949 is written correctly
            if sell_amount > 0:
                assert form8949[0]['Description'] == f"{round(sell_amount, 8):.8f}" + " " + asset
                assert form8949[0]['Date Acquired'] == f"2024-01-01"
                assert form8949[0]['Date Sold'] == f"2024-04-01"
                assert form8949[0]['Proceeds'] == f"{round(100, 2):.2f}"
                assert form8949[0]['Cost Basis'] == f"{100 - round(fifo[asset][0]['cost'], 2):.2f}"
                assert form8949[0]['Gain or Loss'] == f"{round(100 - (100 - fifo[asset][0]['cost']), 2):.2f}"
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
                                    datetime(2024, 4, 1))
        assert len(fifo[asset]) == 2
        assert fifo[asset][0]['amount'] == pytest.approx(1.00001, rel=0, abs=1e-8)
        assert fifo[asset][0]['price'] == pytest.approx(11, rel=0, abs=1e-6)
        assert fifo[asset][0]['cost'] == pytest.approx(55*1.002*1.00001/5, rel=0, abs=1e-6)
        assert fifo[asset][0]['timestamp'] == datetime(2024, 2, 1)

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
                      'Timestamp': '2024-09-04 00:00:00'})

@pytest.fixture(scope="function")
def row1():
    return pd.Series({'Date': '5 / 22 / 2025', 'Asset': 'NVDA',
                      'Amount (asset)': 10.0, 'Sell price ($)': 'NaN',
                      'Buy price ($)': 12.0, 'Account number': 1234,
                      'Entity': 'Chase', 'Notes': '', 'Remaining': '',
                      'Timestamp': '2024-09-04 00:00:00'})

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
