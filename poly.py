import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

X = 4*np.random.rand(100,1)-2
y = 4 + 5*X**2 + np.random.rand(100,1)

poly_features = PolynomialFeatures(degree=2,include_bias=False)
X_poly = poly_features.fit_transform(X)


reg = LinearRegression()
reg.fit(X_poly,y)

#100 valori da 0 a 1
X_vals = np.linspace(-2,2,100).reshape(-1,1)
X_vals_poly = poly_features.fit_transform(X_vals)

y_vals = reg.predict(X_vals_poly)

plt.scatter(X,y)
plt.plot(X_vals,y_vals,color="red")
plt.show()
