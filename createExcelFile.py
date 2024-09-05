import pandas as pd

# Data for Emily's budget
data = {
    "Category": ["Income", "Housing", "Utilities", "Groceries", "Transportation", "Insurance", 
                 "Entertainment", "Debt Payments", "Savings", "Miscellaneous"],
    "Monthly Amount ($)": [8000, 1000, 300, 500, 400, 200, 150, 1650, 2000, 500]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Calculate annual totals
df['Annual Amount ($)'] = df['Monthly Amount ($)'] * 12

# Add a Total row
total = df[['Monthly Amount ($)', 'Annual Amount ($)']].sum()
total_row = pd.DataFrame(data=[ ["Total", total['Monthly Amount ($)'], total['Annual Amount ($)'] ] ], 
                         columns=df.columns)
df = pd.concat([df, total_row])

# Save the budget to an Excel file
file_path = 'data/Emily_Budget.xlsx'
df.to_excel(file_path, index=False)

file_path
