# данный код обрабатывает изображения формата .tiff для лабораторной работы 4.03

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit


def intensity(image, coord, radius):
    intensity_sum = 0
    pixel_count = 0
    x, y = coord

    for i in range(x - radius, x + radius + 1):
        for j in range(y - radius, y + radius + 1):
            if i >= 0 and i < image.width and j >= 0 and j < image.height:
                intensity_sum += sum(image.getpixel((i, j)))
                pixel_count += 1
    return intensity_sum / pixel_count


def sinc(x, A, w):
    return A * np.abs(np.sin(np.pi * x ** 2 * w) / (np.pi * x ** 2 * w))


image = Image.open("/home/danil/PycharmProjects/lab4.03/photos/yellow.tiff")

# выбор центра окружности (x, y), настраивается вручную

center = (591, 957)

l = 2.1773  # мкм/пкс
l = l * 1e-6

# задать отрезок для анализа в пикселях
X_d = range(center[0], 2015)
n = len(X_d)
R_d = np.zeros(n)
for i in range(0, n):
    R_d[i] = abs(center[0] - X_d[i])
    R_d[i] = R_d[i] * l

# создаём массив под интенсивность в каждом пикселе по x в отрезке
I = np.zeros(n)

# Вычисление интенсивности на отрезке X_d
# Радиус -- радиус вокруг точки в пикселях для подсчёта интенсивности рядом, произвольная вещь
# но для лучшего сглаживания попросим побольше. Хотя слишком много тоже плохо.
r = 20

for i in range(0, n):
    I[i] = intensity(image, (X_d[i], center[1]), r)

plt.plot(R_d, I, c='r', lw=1.2, label='I (r)')

# найдём все локальные экстремумы чуть дальше центра
dist = 150
I = np.array(list((I.tolist())))
R_d = np.array(list((R_d.tolist())))
peaks, _ = find_peaks(I[dist:])
min_peaks, _ = find_peaks(-I[dist:])

# отмечаем локальные максимумы и минимумы на графике
plt.plot(R_d[dist:][peaks], I[dist:][peaks], 'bo')
plt.plot(R_d[dist:][min_peaks], I[dist:][min_peaks], 'go')

plt.legend(loc='best')
plt.xlabel(f'r, м', fontsize=12)
plt.ylabel(r'I, у.е.', fontsize=12)
plt.show()

# посчитаем функцию видности
n = min(len(peaks), len(min_peaks))
peaks = peaks[:n]
min_peaks = min_peaks[:n]
V = np.zeros(n)
for i in range(n):
    V[i] = (I[dist:][peaks][i] - I[dist:][min_peaks][i]) / (I[dist:][peaks][i] + I[dist:][min_peaks][i])

plt.scatter(R_d[dist:][peaks], V, c='b', label='V')

# аппроксимация данных кардинальным синусом
popt, pcov = curve_fit(sinc, R_d[dist:][peaks], V, maxfev=2000000000, bounds=([0, -np.inf], [1, np.inf]))
a, w = popt[0], popt[1]

# создание кривой аппроксимации
x = np.linspace(R_d[dist:][peaks][:n][0], R_d[dist:][peaks][:][-1], 5000)
curve_fit = sinc(x, *popt)
plt.plot(x, curve_fit, color='red', label=f'{a:.3f}*|sinc({w:.0f}*r^2)|')

plt.xlabel(f'r, м', fontsize=12)
plt.ylabel(r'V, у.е.', fontsize=12)

plt.legend(loc='best')
plt.show()
