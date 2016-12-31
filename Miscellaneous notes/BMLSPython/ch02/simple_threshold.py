from sklearn.datasets import load_iris

data = load_iris()
features = data['data']
target = data['target']
target_names = data['target_names']
labels = target_names[target]
plength = features[:, 2]

is_setosa = (labels == 'setosa')

max_setosa = plength[is_setosa].max()
min_setosa = plength[~is_setosa].min()

pri
