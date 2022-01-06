import numpy as np
import math
import random

from numpy.core.fromnumeric import size

DATA_PATTERN = "@data"
INPUT_PATTERN = "@inputs"


class KeelData():
  """
  Class responsible for the manipulation, processing and 
  storage of the information contained in the KEEL datasets.
  """
  
  def __init__(self, fh):

    #Place all lines of the file in 'all_lines'without \n.
    all_lines = [item.replace('\n', "") if '\n' in item \
      else item for item in fh.readlines()]
  
    self.attribute_list = self.__getAttLines(all_lines)
    
    # +1 To consider the predicted feature.
    self.number_of_attributes = len(self.attribute_list)+1
    
    self.data_matrix = self.__getData(all_lines)

  def __getData(self, al):
    """
      Extract the data that will be used to train/test the KNN and put it in a matrix.
    """
    line_start = al.index(DATA_PATTERN)
    
    all_rows = al[line_start+1:]
    
    final_matrix = np.zeros(shape=(0, self.number_of_attributes), dtype='O')
    rows = []
    for item in all_rows:
      row = np.array(item.replace(" ", "").split(',')).astype('O')
      rows.append(row)

    final_matrix = np.vstack([rows, final_matrix])

    return final_matrix
    
  def __getAttLines(self, al):
    """
      Get the name of all the attributes in the file dataset.
    """
    item_list = list()
    for item in al:
      if INPUT_PATTERN in item:
        ni = item.split(INPUT_PATTERN)[1]
        
        ni = ni.replace("\n", "").replace(" ", "")
        item_list = ni.split(',')
        break
  
    return item_list
        
  def trainTestSplit(self, train_proportion=0.7):
    """
    Split the file dataset in two, w.r.t train_proportion attribute. The test proportion is 1-train_proportion.
    Returns three parameters, in the form: train_matrix,test_matrix,test_matrix_hidden.
    
    Return
    -------
    train_matrix => The train data, which contains all original attributes.
    
    test_matrix => The test data, which contains all original attributes excepting the target attributed that will be predicted
    with the KNN method.
    
    test_matrix_hidden => Contains the original classification of the test matrix data for correction.
    """
    if not 0 < train_proportion < 1:
      raise ValueError("Train proportion must be a value between 0 and 1.") 
    
    shuffled_matrix = random.sample(list(self.data_matrix), len(self.data_matrix))
    shuffled_matrix = np.array(shuffled_matrix)
    
    
    n_of_col = np.size(shuffled_matrix,1)
    
    n_of_train_lines = math.ceil(train_proportion*len(shuffled_matrix))
    
    train_mat = shuffled_matrix[:n_of_train_lines, :]
    
    test_mat = shuffled_matrix[n_of_train_lines:, :n_of_col-1]    
    
    answer = shuffled_matrix[n_of_train_lines:, n_of_col-1]
    
    answer = np.array([answer]).T  
    
    return train_mat, test_mat, answer
    
if __name__ == "__main__":
  pass
    