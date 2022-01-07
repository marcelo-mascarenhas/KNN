import argparse

from numpy.lib.shape_base import split
from src.knn import Knn
from src.data import KeelData


def parser():
  """
  Declares the program's helper and argument reader.
  """
  
  parser = argparse.ArgumentParser(description="Program that receives a dataset from KEEL and split it in 70-30 train/test proportion \
  to use the implemented KNN.")
  
  parser.add_argument("-inf", "--input_file", help="Path to the dataset file.", required=True)
  parser.add_argument("--k", help="Number of nearest neighbours for K-NN.", required=True)

  argument = parser.parse_args()
  
  file = argument.input_file
  k = argument.k

  k = int(k)

  return file, k

def main():  

  file, number_of_neighbours = parser()
    
  file_handler = open(file, 'r')
  
  dataset = KeelData(file_handler)
  
  train_data, test_points, test_answer = dataset.trainTestSplit()
  
  tree = Knn(train_data)
  
  prediction = tree.classify(test_points, k=number_of_neighbours)
  
  prec, recall, acc = tree.getMetrics(prediction, test_answer)
  
  
  name = file.split('.')[0].split('/')[1]
  print(f'{name}&{round(acc,7)}&{round(prec,2)}&{round(recall,2)}&{number_of_neighbours}')
  
  
  
  
if __name__ == "__main__":
  main()