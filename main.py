import sys
import argparse
import src.knn
from src.data import Data


def parser():
  """
  Declares the program's helper and argument reader.
  """
  
  
  parser = argparse.ArgumentParser(description="Program that receives a dataset from KEEL and split it in 70-30 train/test proportion \
  to use the implemented KNN.")
  
  parser.add_argument("-inf", "--input_file", help="Path to the dataset file.",required=True)

  argument = parser.parse_args()
  
  file = argument.input_file

  return file


def main():  
  
  file = parser()
  
  file_handler = open(file, 'r')
  
  dataset = Data(file_handler, split=True)
  

if __name__ == "__main__":
  main()