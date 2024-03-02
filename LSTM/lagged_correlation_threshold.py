import pandas as pd
import tkinter as tk
from tkinter import filedialog
import os
from datetime import datetime

# from tkinter import font as tkfont

# Choose the two CSV files to test for, named A and B, and choose the duration of the lag and the output file name. Display these with a single GUI window and make to import the required libraries.


def main():
    def select_file(var):
        filename = filedialog.askopenfilename()
        var.set(filename)

    def validate_lag(P):
        if P.isdigit() or P == "":
            return True
        else:
            return False

    def calculate_lagged_correlation(spy_path, vix_path, lag, output_file, low_thres=None, high_thres=0):

        if not os.path.exists(spy_path) or not os.path.exists(vix_path) or lag < 0:
            return

        # Drop timezones, give proper names, and add change column
        spy_df = pd.read_csv(spy_path)
        spy_df.rename(columns={"Date": "Time"}, inplace=True)
        spy_df["Time"] = spy_df["Time"].str[:-6]
        spy_df["Change"] = spy_df["Close"][::-1].diff()
        spy_df["%Chg"] = spy_df["Close"][::-1].pct_change() * 100
        spy_df = spy_df[:-1]

        vix_df = pd.read_csv(vix_path)
        vix_df.rename(columns={"Date": "Time"}, inplace=True)
        vix_df["Time"] = vix_df["Time"].str[:-6]
        vix_df["Change"] = vix_df["Close"][::-1].diff()
        vix_df["%Chg"] = vix_df["Close"][::-1].pct_change() * 100
        vix_df = vix_df[:-1]

        start = datetime(2016, 1, 1)
        end = datetime(2021, 12, 1)

        if low_thres is None:
            low_thres = high_thres
        # SPY

        # Split Time into Date and Time
        spy_df['Date'] = spy_df['Time'].str.split(' ').str[0]

        # Convert Date from M/D/YY to M/D/YYYY
        spy_df['Date'] = pd.to_datetime(spy_df['Date'])

        # Apply date range
        spy_df = spy_df[(spy_df['Date'] > start) & (spy_df['Date'] < end)]

        spy_df['Time'] = spy_df['Time'].str.split(' ').str[1]

        # convert time to datetime
        spy_df['Time'] = pd.to_datetime(spy_df['Time'], format='%H:%M:%S')

        spy_change = spy_df[['Date', 'Time', 'Change', '%Chg']]

        # Convert %Chg to float
        spy_change['%Chg'] = spy_change['%Chg'].astype(float)

        # convert to pivot table
        spy_change_pivot = spy_change.pivot(
            index='Date', columns='Time', values='%Chg')
        spy_vals_pivot = spy_change.pivot(
            index='Date', columns='Time', values='%Chg')

        # convert values to -1 or 1, set NaN for 0
        spy_change_pivot = spy_change_pivot.applymap(
            lambda x: 1 if x >= high_thres else -1 if x <= -low_thres else pd.NA)

        # VIX
        vix_df['Date'] = vix_df['Time'].str.split(' ').str[0]

        # Convert Date from MM/SD/YY to M/D/YYYY
        vix_df['Date'] = pd.to_datetime(vix_df['Date'])

        # Apply date range
        vix_df = vix_df[(vix_df['Date'] > start) & (vix_df['Date'] < end)]

        vix_df['Time'] = vix_df['Time'].str.split(' ').str[1]

        # convert time to datetime
        vix_df['Time'] = pd.to_datetime(vix_df['Time'], format='%H:%M:%S')

        vix_change = vix_df[['Date', 'Time', 'Change', '%Chg']]

        # Convert %Chg to float
        vix_change['%Chg'] = vix_change['%Chg'].astype(float)

        vix_change_pivot = vix_change.pivot(
            index='Date', columns='Time', values='%Chg')
        vix_vals_pivot = vix_change.pivot(
            index='Date', columns='Time', values='%Chg')

        vix_change_pivot = vix_change_pivot.applymap(
            lambda x: 1 if x > 0 else -1)

        # For the two pivot tables, keep only the Date rows that are in both tables
        valid_rows = spy_change_pivot.index.intersection(
            vix_change_pivot.index)

        # index both pivot tables with the valid rows
        spy_change_pivot = spy_change_pivot.loc[valid_rows]
        vix_change_pivot = vix_change_pivot.loc[valid_rows]

        probs_df = pd.DataFrame(
            columns=['a_down_b_down',  'a_down_b_up', 'a_up_b_down', 'a_up_b_up'])
        dist_df = pd.DataFrame(
            columns=['a_down_b_down',  'a_down_b_up', 'a_up_b_down', 'a_up_b_up'])

        # For each 15 minute range, calculate the conditional probabilities
        for i in range(len(spy_change_pivot.columns) - 1):
            spy_col = spy_change_pivot.columns[i]

            # get the column 15 minutes later
            fifteen_minutes_later = spy_col + pd.Timedelta(minutes=lag)

            # if vix_change_pivot doesn't have the column, skip it
            if fifteen_minutes_later not in vix_change_pivot.columns:
                continue

            vix_col = fifteen_minutes_later

            # generate name based on column names. Have name only include the HH:MM part of the time
            name = f"{spy_col.strftime('%H:%M')}_{vix_col.strftime('%H:%M')}"

            # get the column from spy_change_pivot
            spy_col = spy_change_pivot[spy_col]
            vix_col = vix_change_pivot[vix_col]

            # get the combinations in tuple pairs, and any pairs with NaN are dropped
            combo_col = pd.DataFrame(list(zip(spy_col, vix_col))).dropna()

            # print(combo_col)

            # calculate conditional probabilities.
            # from https://stackoverflow.com/questions/37818063/how-to-calculate-conditional-probability-of-values-in-dataframe-pandas-python
            spy_prob = combo_col.groupby(0).size().div(len(combo_col))

            conditional = combo_col.groupby([0, 1]).size().div(
                len(combo_col)).div(spy_prob, axis=0, level=0)

            # convert to new dataframe in the format (index 0, index 1): value
            conditional = conditional.unstack().fillna(0).to_dict()

            full_conditional = {}

            full_conditional_stats = {}

            rename = {-1: "down", 1: "up"}

            for i in [-1, 1]:
                for j in [-1, 1]:
                    if i not in conditional or j not in conditional[i]:
                        full_conditional[f"a_{rename[j]}_b_{rename[i]}"] = 0
                        full_conditional_stats[f"a_{rename[j]}_b_{rename[i]}"] = [
                            0, 0, 0]

                    else:
                        full_conditional[f"a_{rename[j]}_b_{rename[i]}"] = conditional[i][j]

                        # get the rows for each combination of a and b
                        a_down_b_down = combo_col[(
                            combo_col[0] == j) & (combo_col[1] == i)]

                        # get the count, mean, and std from vix_vals_pivot for the rows in a_down_b_down for the hour
                        a_down_b_down_vals = vix_vals_pivot.iloc[a_down_b_down.index].describe().fillna(
                            0)[[fifteen_minutes_later]].to_dict()[fifteen_minutes_later]

                        full_conditional_stats[f"a_{rename[j]}_b_{rename[i]}"] = [int(
                            a_down_b_down_vals['count']), a_down_b_down_vals['mean'], a_down_b_down_vals['std']]

                        full_conditional_stats

            # # add the percentages to the dataframe using the name as the index
            probs_df.loc[name] = full_conditional
            dist_df.loc[name] = full_conditional_stats
            # print("----------")

        # get base file name without extension
        output_file_base = os.path.splitext(output_file)[0]

        probs_df.to_csv(output_file)
        dist_df.to_csv(f"{output_file_base}_stats.csv")

        # print full output path
        print(f"Output file: {os.path.abspath(output_file)}")

    # def check_fields(*args):
    #     if file_a_var.get() and file_b_var.get() and lag_var.get() and output_file_var.get():
    #         generate_button.config(state=tk.NORMAL)
    #     else:
    #         generate_button.config(state=tk.DISABLED)

    root = tk.Tk()
    root.withdraw()
    root.wm_attributes('-topmost', 1)

    file_a = filedialog.askopenfilename(title="Select CSV A")
    print(f"File A: {file_a}")
    file_b = filedialog.askopenfilename(title="Select CSV B")
    print(f"File B: {file_b}")
    lag = input("Enter the lag: ")
    output_file = input("Enter the output file name: ")

    low_thresh = abs(float(input("Enter the low threshold: ")))
    high_thresh = abs(float(input("Enter the high threshold: ")))

    calculate_lagged_correlation(file_a, file_b, int(
        lag), output_file, low_thresh, high_thresh)

    root.destroy()
    # root = tk.Tk()
    # root.title("Lagged Correlation")
    # root.geometry("550x120")  # Increase window size

    # # Font configuration
    # default_font = tkfont.nametofont("TkDefaultFont")
    # default_font.configure(size=12)
    # root.option_add("*Font", default_font)

    # file_a_var = tk.StringVar()
    # file_b_var = tk.StringVar()
    # lag_var = tk.StringVar()
    # output_file_var = tk.StringVar()

    # file_a_var.trace("w", check_fields)
    # file_b_var.trace("w", check_fields)
    # lag_var.trace("w", check_fields)
    # output_file_var.trace("w", check_fields)

    # vcmd = root.register(validate_lag)

    # # First row
    # # First row
    # file_a_button = tk.Button(root, text="Select File A", command=lambda: select_file(file_a_var))
    # file_a_button.grid(row=0, column=0, padx=5, pady=5)

    # file_a_entry = tk.Entry(root, textvariable=file_a_var, width=10)
    # file_a_entry.grid(row=0, column=1, padx=5, pady=5)

    # file_b_button = tk.Button(root, text="Select File B", command=lambda: select_file(file_b_var))
    # file_b_button.grid(row=0, column=2, padx=5, pady=5)

    # file_b_entry = tk.Entry(root, textvariable=file_b_var, width=10)
    # file_b_entry.grid(row=0, column=3, padx=5, pady=5)

    # lag_label = tk.Label(root, text="Enter Lag:")
    # lag_label.grid(row=1, column=0, padx=5, pady=5)

    # lag_entry = tk.Entry(root, textvariable=lag_var, validate="key", validatecommand=(vcmd, '%P'), width=10)
    # lag_entry.grid(row=1, column=1, padx=5, pady=5)

    # # Second row
    # output_file_label = tk.Label(root, text="Output File Name:")
    # output_file_label.grid(row=1, column=2, padx=5, pady=5)

    # output_file_entry = tk.Entry(root, textvariable=output_file_var, width=10)
    # output_file_entry.grid(row=1, column=3, columnspan=3, padx=5, pady=5)

    # generate_button = tk.Button(root, text="Generate CSV", command=calculate_lagged_correlation, state=tk.DISABLED)
    # generate_button.grid(row=2, column=1, columnspan=2, padx=5, pady=5, sticky="ew")

    # check_fields()

    # root.mainloop()


if __name__ == "__main__":
    main()
