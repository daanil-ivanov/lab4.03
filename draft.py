import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Генерация случайных данных
np.random.seed(0)
x = np.linspace(0, 10, 100)
y = 2 * np.sin(1.5 * x) + np.random.normal(size=x.size)

# Функция кардинального синуса
def cardinal_sine(x, A, w, phi):
    return A * np.sinc(w * (x - phi))

# Аппроксимация данных кардинальным синусом
popt, pcov = curve_fit(cardinal_sine, x, y)
print(np.diag(popt))
# Создание кривой аппроксимации
curve_fit = cardinal_sine(x, *popt)

# Построение графиков
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='blue', label='Исходные точки')
plt.plot(x, curve_fit, color='red', label='Аппроксимация кардинальным синусом')
plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Аппроксимация случайных данных кардинальным синусом')
plt.show()
