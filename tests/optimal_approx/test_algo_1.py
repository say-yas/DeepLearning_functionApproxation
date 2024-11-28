import unittest
import pandas as pd
import math


df = pd.read_csv('tests/optimal_approx/table_1.csv')

class TestOptimalApproximation(unittest.TestCase):

    def run_test_for_function(self, func_name):
        for _, row in df[df['Function'] == func_name].iterrows():
            segments = row['n']
            delta_s = row['Mean ΔS']
            paper_rmse = row['paper_rmse']
            
            with self.subTest(f"{func_name} function with {segments} segments"):
                # Compare Delta S with paper's RMSE
                self.assertAlmostEqual(delta_s, paper_rmse, places=3, 
                                       msg=f"Delta S differs from paper's RMSE for {func_name} function with {segments} segments")
                
                # Optional: Print the comparison results
                print(f"{func_name} with {segments} segments: Delta S = {delta_s}, Paper RMSE = {paper_rmse}")

    def test_exponential_function(self):
        self.run_test_for_function('e(x)')

    def test_square_function(self):
        self.run_test_for_function('x**2')

    def test_cubic_function(self):
        self.run_test_for_function('x**3')

if __name__ == '__main__':
    unittest.main()