from numpy.core.defchararray import count
from .kdtree import *
from sortedcontainers import SortedDict

class Knn(Kdtree):
  
  def __init__(self, train_points):
    
    super().__init__(train_points)

  def __findKNearestNeighbours(self, point, k):
    
    if np.size(point, 0) != self.dimension_num:
      raise ValueError("Test point dimension does not match KDTree's.")
    
    
    # target_node = self.__findTargetNode(point, k)    
    # candidate_points = list()
    # self.__getCandidatePoints(target_node, candidate_points)
    
    # nearest_points = self.__getNearestPoints(point, candidate_points, k )    
    
    nearest = SortedDict()
    dimension = 0
    self.__nearestNeighboursAux(point, self.kdtree, k, nearest, dimension)
    nearest_points = list(nearest.values())
    
    return nearest_points
  
  def __nearestNeighboursAux(self, point, node, k, nearest, dimension):

    median_comparator = float(point[dimension])

    if self.__isNode(node): 
      
      if median_comparator <= node.median_value:
        good_branch = node.left
        bad_branch = node.right
      else:
        good_branch = node.right
        bad_branch = node.left

      self.__nearestNeighboursAux(point, good_branch, k, nearest, self._Kdtree__checkDimensions(dimension+1))
      ## The radius is the distance between the farest point and the target point.
      radius = nearest.keys()[-1]
      
      if (median_comparator-radius) <= node.median_value <= median_comparator+radius or len(nearest) != k:
        self.__nearestNeighboursAux(point, bad_branch, k, nearest, self._Kdtree__checkDimensions(dimension+1))
    
    else:
      dist = self.distance(point, node)
      
      if len(nearest) == k and dist < nearest.keys()[-1]: 
        nearest.popitem(k-1)
        nearest[dist] = node
      elif len(nearest) < k:
        nearest[dist] = node



      
  
  def __getNearestPoints(self, point, allCandidates, k):
    
    nearest = SortedDict()
    final_index = k-1
    for candidate in allCandidates:
      
      dist = self.distance(point, candidate)
      
      if len(nearest) == k and dist >= nearest.keys()[-1]:
        continue
      elif len(nearest) == k and dist < nearest.keys()[-1]:
        nearest.popitem(final_index)
        
      nearest[dist] = candidate
    
    return list(nearest.values())
        
  def __getCandidatePoints(self, targetNode, lista):    
    
    if self.__isNode(targetNode):
      self.__getCandidatePoints(targetNode.left, lista)
      self.__getCandidatePoints(targetNode.right, lista)
    else:
      lista.append(targetNode)
  
  def __findTargetNode(self, point, k):
    
    target_node = self.kdtree
    
    dimension = 0

    while True:
            
      median_comparator = float(point[dimension])
      
      candidate_node = target_node.left if median_comparator <= target_node.median_value \
        else target_node.right
      
      if not self.__isNode(candidate_node):
        if k == 1:
          target_node = candidate_node
        break
      else:
        if k <= candidate_node.accessible_points:
          target_node = candidate_node
          dimension = self._Kdtree__checkDimensions(dimension+1)
        else:
          break

    
    return target_node
  
  def __checkLabels(self, points):
    counter_dic = dict()
    
    for label, weight in points:
      
      if label not in counter_dic:
        counter_dic[label] = int(weight)
      else:
        counter_dic[label] += int(weight)
  
    highest_label = sorted(counter_dic, key=lambda x: counter_dic[x])[-1]
    return highest_label
  
  def distance(self, point1, point2):
    distance = 0 
    
    candidate_point = point2[:, :self.dimension_num]
    
    distance = np.sum((point1.astype('float32')-candidate_point.astype('float32'))**2)
    
    return math.sqrt(distance)

  def __isNode(self, node):
    
    return True if isinstance(node, self.Node) else False
  

  def classify(self, test_point, k):
    classifications = list()
    
    for item in test_point:
      nearest_points = self.__findKNearestNeighbours(item, k)
      classification_columns = np.concatenate(nearest_points, axis=0)[: , self.dimension_num:]
      result = self.__checkLabels(classification_columns)
      classifications.append(result)
    

    column_prediction = np.array([classifications]).T
    return column_prediction
  
  def getMetrics(self, predictions, answers):
    unique_rows = np.unique(answers.astype('<U30'), axis=0)
    first_class = unique_rows[0]
    
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for c1,c2 in zip(predictions, answers):
      if c1 == first_class and c2 == first_class:
        tp +=1
      elif c1 == first_class and c2 != first_class:
        fp +=1
      elif c1 != first_class and c2 == first_class:
        fn +=1
      else:
        tn +=1
        
    accuracy = (tp+tn)/(tp+tn+fp+fn)
    
    prec = tp/(tp+fp)
    recall = tp/(tp+fn)
    
    return accuracy, prec, recall
  
    
      
if __name__ == "__name__":
  pass