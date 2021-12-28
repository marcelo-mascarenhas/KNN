import numpy

DATA_PATTERN = "@data\n"
INPUT_PATTERN = "@inputs"


class Data():
  
  def __init__(self, fh, split=False):

    all_lines = fh.readlines()
    
    self.attribute_list = self.__getAttLines(all_lines)
    
    
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
   
    