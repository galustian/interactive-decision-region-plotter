from sklearn.ensemble import GradientBoostingClassifier
from plotter import plot_decision_region

clf = GradientBoostingClassifier()
plot_decision_region(clf)
