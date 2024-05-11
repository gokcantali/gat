def get_data_insights(df, file_path):
    with open(file_path, 'w') as f:
        f.write("\nData info:\n")
        f.write("----------------------------------------------\n")
        df.info(buf=f)

        f.write("\nData describe:\n")
        f.write("----------------------------------------------\n")
        f.write(df.describe().to_string())

        f.write("\nData nunique:\n")
        f.write("----------------------------------------------\n")
        f.write(df.nunique().to_string())

        f.write("\nData correlation:\n")
        f.write("----------------------------------------------\n")
        f.write(df.corr().to_string())

        f.write("\nData skew:\n")
        f.write("----------------------------------------------\n")
        f.write(df.skew().to_string())

        f.write("\nData kurt:\n")
        f.write("----------------------------------------------\n")
        f.write(df.kurt().to_string())
