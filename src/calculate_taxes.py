# Utility to record a sale
def record_sale(token, amount, proceeds, cost_basis, acquisition_date, sale_date):
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