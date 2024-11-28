import unittest
import pandas as pd
import math

# Read the dataframe from the saved file
# Replace 'path_to_your_file.csv' with the actual path to your saved dataframe
df = pd.read_csv('tests/optimal_approx/table_1.csv')

class TestApproximationAlgorithms(unittest.TestCase):

    def compare_rmse(self, calculated_rmse, paper_rmse):
        # Check if the first 3 decimal places are equal
        if round(calculated_rmse, 3) == round(paper_rmse, 3):
            return True
        # If not, check if calculated RMSE is less than paper RMSE
        return calculated_rmse < paper_rmse

    def run_test_for_function(self, func_name):
        for _, row in df[df['Function'] == func_name].iterrows():
            segments = row['n']
            delta_s = row['Mean ΔS']
            paper_rmse = row['paper_rmse']
            imamototang_rmse = row['ammamoto_rmse']
            imamototang_paper_rmse = row['ammamoto_paper_rmse']
            
            with self.subTest(f"Original algorithm - {func_name} function with {segments} segments"):
                self.assertTrue(self.compare_rmse(delta_s, paper_rmse), 
                                f"Delta S ({delta_s}) is not better than or equal (to 3 decimal places) to paper's RMSE ({paper_rmse}) for {func_name} function with {segments} segments")
            
            with self.subTest(f"Imamototang algorithm - {func_name} function with {segments} segments"):
                if imamototang_paper_rmse != 'unconverged':
                    self.assertTrue(self.compare_rmse(float(imamototang_rmse), float(imamototang_paper_rmse)), 
                                    f"Imamototang RMSE ({imamototang_rmse}) is not better than or equal (to 3 decimal places) to paper's RMSE ({imamototang_paper_rmse}) for {func_name} function with {segments} segments")
                else:
                    print(f"Skipping Imamototang test for {func_name} function with {segments} segments due to unconverged paper result")

            # Optional: Print the comparison results
            print(f"{func_name} with {segments} segments:")
            print(f"  Original: Delta S = {delta_s}, Paper RMSE = {paper_rmse}")
            print(f"  Imamototang: RMSE = {imamototang_rmse}, Paper RMSE = {imamototang_paper_rmse}")

    def test_exponential_function(self):
        self.run_test_for_function('e(x)')

    def test_square_function(self):
        self.run_test_for_function('x**2')

    def test_cubic_function(self):
        self.run_test_for_function('x**3')

if __name__ == '__main__':
    unittest.main()