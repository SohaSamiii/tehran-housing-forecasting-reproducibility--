# Data

`raw/Home_4.xlsx` is the workbook supplied for this study. The analysis reads the `Sheet2` worksheet.

The source sheet contains 1,958 district-month records: 22 Tehran municipal districts observed over 89 months. Required columns are:

| Source column | Analysis name | Description |
|---|---|---|
| YEAR | `year` | Jalali year |
| MONTH | `month` | Jalali month (1–12) |
| REGION | `district` | Tehran district identifier (1–22) |
| AVERAGE PRICE | `price` | Nominal average housing price per square metre, source units |
| SHAKHES MASALEH | `mas` | Construction-material price index |
| TAVAROM MASKAN | `tav` | Housing inflation index |
| SHAKHES BORS | `bors` | Tehran Stock Exchange index |
| USD | `usd` | US-dollar exchange-rate series |
| GOLD | `gold` | Gold-price series |

The workbook is included to make the review package executable without an additional download. Before making the repository public, the authors should verify that they have the right to redistribute the workbook. The workbook itself does not provide a data licence or a complete source citation. Do not assume that inclusion in this package grants redistribution rights.

If the data cannot be redistributed, remove `raw/Home_4.xlsx` from the public repository and provide reviewers with an authorised access route. The scripts accept a different workbook path through `--data`.
