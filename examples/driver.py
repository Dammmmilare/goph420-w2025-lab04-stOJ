import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))


from src.lab_04.regression import multi_regress

def main():
    y = ... # enter dependent variable data (or load from a file)
    Z = ... # enter independent variable data (or load from a file)
    a, e, rsq = multi_regress(y, Z)

if __name__ == "__main__":
    main()