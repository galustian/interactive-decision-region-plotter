from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from plotter import plot_decision_region

clf = SVC()
plot_decision_region(clf)
