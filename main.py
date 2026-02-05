import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

p_data = np.array([2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0])
f_data = np.array([475, 522, 588, 658, 720, 787.5, 833, 868.5, 898, 922, 946.5, 971, 997, 1013.5, 1042.5, 1066.5, 1083, 1097.5, 1120, 1136, 1163.5, 1168, 1178.5, 1197, 1214.5, 1232.5, 1247, 1264, 1271.5, 1292.5, 1303, 1319, 1327, 1332.5, 1338, 1344.5, 1347, 1353.5, 1357.5, 1361, 1367 ])

X = f_data.reshape(-1, 1)
y = p_data

degree = 2

poly = PolynomialFeatures(degree)
X_poly = poly.fit_transform(X)

model = LinearRegression()
model.fit(X_poly, y)

y_pred = model.predict(X_poly)

params = [model.intercept_, *model.coef_[1:]]

print("Degrees", degree, end="\n")
for p in params:
    print(p, end="\t")
print()

r2 = r2_score(y, y_pred)
print(f"R^2 score: {r2:.6f}")

plt.scatter(f_data, p_data)

plt.plot(f_data, model.predict(X_poly), color='red')

plt.title('Polynomial Regression')
plt.xlabel('FSR')
plt.ylabel('Pressure')

plt.show()