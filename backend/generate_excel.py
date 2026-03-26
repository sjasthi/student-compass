import pandas as pd

def generate_excel():

    # load csv file
    df = pd.read_csv("rag_results.csv")

    # export to excel
    df.to_excel("rag_results_conclusion.xlsx", index=False)

    print("Excel file generated: rag_results_conclusion.xlsx")

if __name__ == "__main__":
    generate_excel()