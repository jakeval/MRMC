from typing import Optional
import pandas as pd
from data.datasets import base_loader
from data.datasets import utils


DATASET_NAME = "credit_card_default"
_SOURCE_TARGET_COLUMN = "default payment next month"
_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/"
    "default%20of%20credit%20card%20clients.xls"
)


class CreditCardDefaultLoader(base_loader.DataLoader):
    """Loads the UCI Credit Card Default dataset from the internet or local
    storage.

    The dataset has columns:
        LIMIT_BAL: The amount of the given credit.
        SEX: Binary variable where 1 is Male and 2 is Female.
        EDUCATION: A categorical variable. The dataset source includes
            descriptions for values 1, 2, 3, and 4, but does not explain values
            0, 5, and 6 which are also included in the data. The column is
            dropped for this reason.
        MARRIAGE: A categorical variable. The dataset source includes
            descriptions for values 1, 2, and 3, but does not explain value 0
            which is also included in the data. The column is dropped for this
            reason.
        AGE: Numerical age in years.
        PAY_[1-6]: The columns PAY_1, PAY_2, ..., PAY_6 correspond to
            information about the past 6 months. Values -2, -1, and 0 indicate
            that the bill was paid on on time. Values 1 through 9 indicate how
            many months late the payment was.
        BILL_AMT[1-6]: The columns BILL_AMT1, BILL_AMT2, ..., BILL_AMT6
            correspond to information about the past 6 months. Each column
            value indicates the amount in dollars of the credit card bill for
            that month.
        PAY_AMT[1-6]: The columns PAY_AMT1, PAY_AMT2, ..., PAY_AMT6
            correspond to information about the past 6 months. Each column
            value indicates the amount in dollars that the customer paid for
            that month."""

    def __init__(
        self, only_continuous_vars: bool = True, data_dir: Optional[str] = None
    ):
        """Creates a data loader for the UCI Credit Card Default dataset.

        Args:
            only_continuous_vars: If True, drops the categorical SEX column and
                             turns the categorical PAY_* columns into
                             continuous columns."""
        super().__init__(
            continuous_features=["LIMIT_BAL", "AGE"]
            + [f"PAY_{i}" for i in range(1, 7)]
            + [f"BILL_AMT{i}" for i in range(1, 7)]
            + [f"PAY_AMT{i}" for i in range(1, 7)],
            ordinal_features=[],
            categorical_features=[],
            label_name="Y",
            positive_label=0,
            data_dir=data_dir,
            dataset_name=DATASET_NAME,
        )
        self.only_continuous_vars = only_continuous_vars

    def download_data(self) -> pd.DataFrame:
        return pd.read_excel(_URL, header=1)

    def process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Processes the Credit Card Default dataset.

        Drops the MARRIAGE and EDUCATION columns. Renames the PAY_0 column to
        PAY_1. if self.only_continuous_vars is True, drops the SEX column and
        sets the values (0, -1, -2) of the PAY_* columns to 0 to make the
        column continuous instead of categorical.

        Args:
            data: The data to process.

        Returns:
            A processed DataFrame."""
        data = (
            data.set_index("ID")
            .drop(columns=["MARRIAGE", "EDUCATION"])
            .rename(columns={_SOURCE_TARGET_COLUMN: "Y", "PAY_0": "PAY_1"})
        )
        if self.only_continuous_vars:
            pay_mapping = {0: [-1, -2]}
            for pay_column in [f"PAY_{i}" for i in range(1, 7)]:
                data[pay_column] = utils.recategorize_feature(
                    data[pay_column], pay_mapping
                )
            data = data.drop(columns=["SEX"])
        return data
