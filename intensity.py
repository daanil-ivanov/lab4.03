# данный код обрабатывает изображения формата .tiff для лабораторной работы 4.03

from PIL import Image
import math
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

def sinc(x, w):
    return 0.76 * np.abs(np.sin(np.pi * x**2 * w) / (np.pi * x**2 * w))

# Загрузка изображения
# голый 1 -- зелёный
# голый 2 -- жёлтый
image = Image.open("/home/danil/PycharmProjects/lab4.03/photos/голый 2.tiff")

# Выбор центра окружности (x, y)
# настраивается руками
# жёлтый (1394, 661)
# зелёный (1399, 671)

center = (1394, 661)

# просто в gnu imp открыл изображение, где нанёс размеры заранее
# и посчитал расстояние в пикселях для нанесённого отрезка.
pix_len = 1971-1644
m_len = 3e+03 * 1e-6 # уже в метрах

l = m_len/pix_len # длина одного пикселя в метрах без фактора

# так как при снятии лабы я криво указал длины (не тот размер щелей между ними оказался),
# то я домножаю на фактор 1/3, так как похоже, что в 3 раза ошибаюсь
l *= 1/3

# Задать отрезок для анализа в пикселях
# ж 12, з 37
X_d = range(12, center[0])
n = len(X_d)
R_d = np.zeros(n)
for i in range(0, n):
    R_d[i] = center[0] - X_d[i]
    R_d[i] = R_d[i] * l

# Создаём массив под интенсивность в каждом пикселе по x в отрезке
I = np.zeros(n)

# Вычисление интенсивности на отрезке X_d
# Радиус -- радиус вокруг точки в пикселях для подсчёта интенсивности рядом, произвольная вещь
# в целом не очень важна, хоть 3, хоть 5

# но для лучшего сглаживания попросим побольше, поставил 21 для Ж. Слишком много тоже плохо.
r = 21

for i in range(0, n):
    I[i] = intensity(image, (X_d[i], center[1]), r)

# получается, что к краям уходит ниже, чем в центре, что непорядок
# скажем, что в центре условный ноль, а что ниже по модулю отразмим от нуля

abs_null = min(I[-40:-1])
for i in range(0, n):
    I[i] = abs((I[i] - abs_null))

plt.plot(R_d, I, c='r', lw=1.2, label='I (r)')



# найдём все локальные экстремумы чуть дальше центра
dist = 110
I = np.array(list(reversed(I.tolist())))
R_d = np.array(list(reversed(R_d.tolist())))
# Находим локальные максимумы
peaks, _ = find_peaks(I[dist:])
# Находим локальные минимумы
min_peaks, _ = find_peaks(-I[dist:])

# Отмечаем локальные максимумы и минимумы на графике
plt.plot(R_d[dist:][peaks], I[dist:][peaks], 'ro', label='Локальные максимумы')
plt.plot(R_d[dist:][min_peaks], I[dist:][min_peaks], 'go', label='Локальные минимумы')

# plt.title(f"r = {r}")
plt.legend(loc='best')
plt.show()

# посчитаем функцию видности. Будем считать, что V(r), r -- радиус до максимума
n = min(len(peaks), len(min_peaks))
peaks = peaks[:n]
min_peaks = min_peaks[:n]
V = np.zeros(n)
# print(I[dist:][peaks])
# print(I[dist:][min_peaks])
for i in range(n):
    V[i] = (I[dist:][peaks][i] - I[dist:][min_peaks][i]) / (I[dist:][peaks][i] + I[dist:][min_peaks][i])
    # print(V[i])

plt.scatter(R_d[dist:][peaks], V, c='b', label='V')



# Аппроксимация данных кардинальным синусом
# тут берём первые n штук, далее, как по теории и как видно, видимость начинает обратно нарастать, что портит наш фит
n = -1

popt, pcov = curve_fit(sinc, R_d[dist:][peaks][:n], V[:n], maxfev=2000000000)#, bounds=([0, -np.inf], [1, np.inf]))
print(np.sqrt(np.diag(np.abs(popt))))
# Создание кривой аппроксимации
x = np.linspace(R_d[dist:][peaks][:n][0], R_d[dist:][peaks][:n][-1], 5000)
curve_fit = sinc(x, *popt)
plt.plot(x, curve_fit, color='red', label='sinc')

# эмпирически подобрали, что A = 0.87 )))) для жёлтого

plt.legend(loc='best')
plt.show()
