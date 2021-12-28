

class Kdtree():
  
  class Node():
    """
    Inner class that builds the foundation to the construction of the KD-tree
    """
    def __init__(self, mv):
      self.median_value = mv
      self.right_node = None
      self.left_node = None
  
  
  
  def __init__(self, points):
    self