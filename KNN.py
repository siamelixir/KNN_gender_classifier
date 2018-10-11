from sklearn import neighbors

n_neighbors = 3

#[age,height,weight,shoe size]

X = [19,183,75,46],[21,153,63,37],[32,154,58,39],[20,190,85,47],[34,170,69,40],[23,178,77,43],[30,167,80,37],[23,163,55,41],[31,195,93,48],[29,174,78,42],[25,185,81,45],[28,169,66,39]

#gender: male/female relative to the measurements
y = ['male','female','female','male','male','female','female','male','female','male','male','female']

for weights in ['uniform', 'distance']:
	clf = neighbors.KNeighborsClassifier(n_neighbors, weights = weights)

	clf.fit(X , y)

prediction = clf.predict([[19,182,60,47]])

print (prediction)