import numpy as np
import statistics

from numpy.core.numeric import count_nonzero

class Kdtree():
  
  class Node():
    """
    Inner class that builds the foundation to the construction of the KD-tree
    """
    def __init__(self, mv = None):
      self.median_value = mv if mv is not None else None      
      self.right_node = None
      self.left_node = None
  
  def __init__(self, data_matrix):
    
    new_matrix = self.__treatDuplicates(data_matrix)
    
    #Calculate the dimension of the Tree. It disregards the last column, because the attribute that will be forecasted 
    #shouldn't be considered in the tree construction. Also, it not consider the added weight column.    
    self.dimension_num = np.size(new_matrix, 1)-2
    
    started_dimension = 0
    
    self.kdtree = self.__buildKDTree(new_matrix, started_dimension)
    
    print(self.kdtree)
            
  def __buildKDTreeRec(self, data_matrix, started_dimension, recursion=False):
    
    if len(data_matrix) == 0 and not recursion:
      raise ValueError('Empty dataset was sent to the KDTree constructor')
    
    elif len(data_matrix) <= 1 and recursion:
      return data_matrix

    median, left_matrix, right_matrix = self.__medianAndMatrices(data_matrix, started_dimension)
        
    head_node = self.Node(median)    
    
    head_node.left_node = self.__buildKDTree(left_matrix, \
      self.__checkDimensions(started_dimension+1),  recursion=True)
    
    head_node.right_node = self.__buildKDTree(right_matrix, \
      self.__checkDimensions(started_dimension+1), recursion=True)
    
    
    return head_node
    
  
  def __buildKDTree(self, data_matrix, started_dimension):
    new_stack = list()

    head = self.Node()
    
    current_node = head
    
    new_stack.append((current_node, data_matrix, started_dimension))
    
    while True:
      current_node, data_matrix, started_dimension = new_stack.pop()
      
      median, lm, rm = self.__medianAndMatrices(data_matrix, started_dimension)
      
      started_dimension = self.__checkDimensions(started_dimension+1)
      
      current_node.median_value = median
      
      if len(lm) <= 1 or self.__checkInfinite(lm):
        current_node.left = lm
      else:
        current_node.left = self.Node()
        new_stack.append((current_node.left,lm, started_dimension))
        
        
      if len(rm) <= 1 or self.__checkInfinite(rm):
        current_node.right = rm
      else:
        current_node.right = self.Node()
        new_stack.append((current_node.right, rm, started_dimension))     
      
      if len(new_stack) == 0:
        break
      
    return head
  
  def __medianAndMatrices(self, data_matrix, dim):
    
    att_median = statistics.median(data_matrix[:, dim].astype('float32'))

    condition = data_matrix[:, dim].astype('float32') < att_median
    
    left_mat, right_mat = self.__splitMatrix(data_matrix, condition)
    
    return att_median, left_mat, right_mat
  
  def __checkDimensions(self, curr_dimension):
    return curr_dimension % self.dimension_num
  
  def __splitMatrix(self, matrix, cond):
    return matrix[cond], matrix[~cond]
  
  def __checkInfinite(self, data_matrix):
    """
    Check the cases when all vectors stay on the same side and the tree construction never stops. 
    If it occurs, then put all vectors in one leaf.
    
    For instance, the case:
    [['5.1' '3.5' '1.4' '0.2' ]
      ['5.1' '3.5' '1.4' '0.3']
      ['5.5' '4.2' '1.4' '0.2']]
      
    When you take the median of any dimension and filter by <, it always stay on the same side of the tree, and the algorithm runs forever.
    If you alter to <= or => or >, it happens with another set of vectors.
    
    """
    
    for dim in range(self.dimension_num):
      
      att_median = statistics.median(data_matrix[:, dim].astype('float32'))
      
      condition = data_matrix[:, dim].astype('float32') < att_median      
      
      if not np.all(condition == False):
        return False
    
    return True
      
  def __treatDuplicates(self, data_matrix):
    """
    Concatenate the same points and add a weight column.
    
    """
    
    
    unique_rows, count = np.unique(data_matrix.astype('<U30'), return_counts=True, axis=0)  
    
    new_matrix = np.array(np.append(unique_rows, np.asmatrix(count).T, axis=1).astype('O'))

    return new_matrix
      
  
if __name__ == "__main__":
  pass