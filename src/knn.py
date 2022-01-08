from .kdtree import *

class Knn(KdTree):
  
  def __init__(self, train_points):
    
    super().__init__(train_points)

  def __findKNearestNeighbours(self, point, k):
    
    if np.size(point, 0) != self.dimension_num:
      raise ValueError("Test point dimension does not match KDTree's.")
    
    nearest = list()
    dimension = 0
    self.__nearestNeighboursAux(point, self.kdtree, k, nearest, dimension)
    nearest_points = [ seq[1] for seq in nearest ]
    
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

      self.__nearestNeighboursAux(point, good_branch, k, nearest, self._KdTree__checkDimensions(dimension+1))
      ## The radius is the distance between the farest point and the target point.
      radius = nearest[-1][0]
      
      if (median_comparator-radius) <= node.median_value <= median_comparator+radius or len(nearest) != k:
        self.__nearestNeighboursAux(point, bad_branch, k, nearest, self._KdTree__checkDimensions(dimension+1))
    
    else:
      dist = self.distance(point, node)
      
      if len(nearest) == k and dist < nearest[-1][0]: 
        nearest.pop()
        nearest.append((dist,node))
        #Sort the list to assure that the last element of the list is the farest.
        nearest.sort(key=lambda x: x[0])
      
      elif len(nearest) < k:
        nearest.append((dist, node))
        nearest.sort(key=lambda x: x[0])

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
    """
    Classify each one of the points.
    
    """
    classifications = list()
    for item in test_point:
      nearest_points = self.__findKNearestNeighbours(item, k)
      classification_columns = np.concatenate(nearest_points, axis=0)[: , self.dimension_num:]

      result = self.__checkLabels(classification_columns)
      classifications.append(result)
    

    column_prediction = np.array([classifications]).T
    return column_prediction
  
  def getMetrics(self, predictions, answers):
    """
    Calculate the confusion matrix and derive the accuracy, precision and recall from it.
    
    
    """
    
    
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
      elif c1 != first_class and c2 != first_class:
        tn +=1
    accuracy = (tp+tn)/(tp+tn+fp+fn)
    
    prec = tp/(tp+fp)
    recall = tp/(tp+fn)

        
    return accuracy, prec, recall
  
    
      
if __name__ == "__name__":
  pass
