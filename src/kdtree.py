import numpy as np
import statistics
import math


class KdTree():
  
  class Node():
    """
    Inner class that builds the foundation to the construction of the KD-tree
    """
    def __init__(self, mv = None):
      
      self.median_value = mv if mv is not None else None
            
      self.right = None
      
      self.left = None
  
  def __init__(self, data_matrix):
    
    new_matrix = self.__treatDuplicates(data_matrix)
    
    #Calculate the dimension of the Tree. It disregards the last column, because the attribute that will be forecasted 
    #shouldn't be considered in the tree construction. Also, it not consider the added weight column.    
    self.dimension_num = np.size(new_matrix, 1)-2
    
    started_dimension = 0
    # self.kdtree = self.__buildKDTreeRec(new_matrix, started_dimension)

    self.kdtree = self.__buildKDTree(new_matrix, started_dimension)
                
                
  # def __buildKDTreeRec(self, data_matrix, started_dimension, recursion=False):
    
  #   if len(data_matrix) == 0 and not recursion:
  #     raise ValueError('Empty dataset was sent to the KDTree constructor')
    
  #   elif len(data_matrix) <= 1 and recursion:
  #     return data_matrix

  #   median, left_matrix, right_matrix = self.__medianAndMatrices(data_matrix, started_dimension)
        
  #   head_node = self.Node(median)    
    
  #   head_node.left = self.__buildKDTreeRec(left_matrix, \
  #     self.__checkDimensions(started_dimension+1),  recursion=True)
    
  #   head_node.right = self.__buildKDTreeRec(right_matrix, \
  #     self.__checkDimensions(started_dimension+1), recursion=True)
    
    
  #   return head_node
    
  
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
      
      if len(lm) <= 1:
        current_node.left = lm
      else:
        current_node.left = self.Node()
                
        new_stack.append((current_node.left,lm, started_dimension))
        
      if len(rm) <= 1:
        current_node.right = rm
      else:
        current_node.right = self.Node()
        new_stack.append((current_node.right, rm, started_dimension))     

      if len(new_stack) == 0:
        break
      
    return head
  
  def __medianAndMatrices(self, data_matrix, dim):
    
    att_median = statistics.median(data_matrix[:, dim].astype('float32'))

    left_mat, right_mat = self.__halfMatrix(data_matrix, dim)
    
    return att_median, left_mat, right_mat
  
  def __checkDimensions(self, curr_dimension):
    return curr_dimension % self.dimension_num

  def __halfMatrix(self, data_matrix, dim):
    """
    If the array is ordened, the median is, by definition, right at the middle. So, it is sufficient just splitting
    the matrix at half.
    """
    data_matrix = data_matrix[np.argsort(data_matrix[:, dim])]
    size_mat = math.ceil(len(data_matrix)/2)
    
    lm = data_matrix[:size_mat, :]
    rm = data_matrix[size_mat:, :]
    
    return lm, rm
      
  def __treatDuplicates(self, data_matrix):
    """
    Concatenate the same points and 
    add a weight column.
    """
    
    unique_rows, count = np.unique(data_matrix.astype('<U30'), return_counts=True, axis=0)  
    
    new_matrix = np.array(np.append(unique_rows, np.asmatrix(count).T, axis=1).astype('O'))

    return new_matrix
  
  
  
if __name__ == "__main__":
  pass