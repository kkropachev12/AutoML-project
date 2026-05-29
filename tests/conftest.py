from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def synthetic_telco_csv(tmp_path: Path) -> Path:
    """Create a small Telco-like dataset with both churn classes."""
    rows = []
    for index in range(40):
        churn = "Yes" if index % 2 == 0 else "No"
        monthly_charge = 80.0 if churn == "Yes" else 40.0
        rows.append(
            {
                "customerID": f"{index:04d}-TEST",
                "gender": "Female" if index % 3 == 0 else "Male",
                "SeniorCitizen": index % 2,
                "tenure": index + 1,
                "Contract": "Month-to-month" if churn == "Yes" else "Two year",
                "PaymentMethod": (
                    "Electronic check" if churn == "Yes" else "Credit card"
                ),
                "MonthlyCharges": monthly_charge,
                "TotalCharges": (
                    "bad-value"
                    if index == 0
                    else str(round((index + 1) * monthly_charge, 2))
                ),
                "Churn": churn,
            }
        )

    csv_path = tmp_path / "telco_customer_churn.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path
