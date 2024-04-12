import numpy as np

def R(r, m, n):
    return (r[m]**2 - r[n]**2)/((m-n)*r[-1])

# номера колец
n = [0, 1, 2, 3]
# первые четыре соответствуют 4 кольцам, предпоследнее -- радиус размытости, последнее -- lambda
# последняя буква названия переменной -- цвет
rr = [726., 937., 1100., 1243., 3033., 630.]
ro = [671., 880., 1040., 1180., 2997., 578.]
rg = [651., 852., 1010., 1143., 2883., 546.]
rb = [575., 780., 933., 1073., 2843., 435.]
colors = [rr, ro, rg, rb]
R_c = [[] for _ in range(4)]
# переведём всё в метры
for i, r in enumerate(colors):
    for j, rr in enumerate(r):
        if j == len(r)-1:
            colors[i][j] *= 1e-9
        else:
            colors[i][j] *= 1e-6

for i, r in enumerate(colors):
    for N in n:
        for M in n:
            if M > N:
                 R_c[i].append(R(r, M, N))
blur_rad = 0
for i in range(len(colors)):
    R_c[i] = np.average(R_c[i])
    blur_rad += colors[i][-2]
    print(f"{R_c[i]:.3f} м")

blur_rad /= 4
print(f"{blur_rad * 1e6:.0f} мкм")
