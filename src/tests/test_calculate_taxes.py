import pytest
from datetime import datetime, timedelta
from src.irs_asset_fifo_calculator import calculate_taxes


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
    return 'TSLA'

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
        assert  form8949[1]["Description"] == "10.00000000 TSLA"
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

        with pytest.raises(TypeError, match="A list object must be passed.  "
                + "Create form8949 list first."):
            calculate_taxes.record_sale(form8949, asset, amount,
                                        proceeds, cost_basis, acquisition_date, sale_date)
