"""Factor52 location classes into six anatomical territories and within-territory classes."""
import numpy as np
TERRITORIES={0:list(range(1,7)),1:list(range(7,18)),2:list(range(18,22)),3:list(range(22,36)),4:list(range(36,45)),5:list(range(45,53))}
CLASS_TO_TERRITORY={c:g for g,cs in TERRITORIES.items() for c in cs}
class HierarchicalExtraTrees:
 def __init__(self,parent,children,classes):self.parent=parent;self.children=children;self.classes_=np.asarray(classes)
 def predict_proba(self,X):
  out=np.zeros((len(X),len(self.classes_)));p=self.parent.predict_proba(X);cols={int(c):i for i,c in enumerate(self.classes_)}
  for j,g in enumerate(self.parent.classes_):
   child=self.children[int(g)];cp=child.predict_proba(X)
   for k,c in enumerate(child.classes_):out[:,cols[int(c)]]=p[:,j]*cp[:,k]
  assert np.allclose(out.sum(1),1);return out
 def predict(self,X):return self.classes_[self.predict_proba(X).argmax(1)]
