import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


X = 4*np.random.rand(100,1)-2
y = 4 + 5*X**2 + np.random.rand(100,1)
#plt.scatter(X,y)
#plt.show()

reg = LinearRegression()
reg.fit(X,y)
#100 valori da 0 a 1
X_vals = np.linspace(-2,2,100).reshape(-1,1)
y_vals = reg.predict(X_vals)

plt.scatter(X,y)
plt.plot(X_vals,y_vals,color="red")
plt.show()
