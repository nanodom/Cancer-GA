import numpy as np

def _ctp_base(x):
    f1 = float(x[0])
    g = float(1.0 + 9.0 * np.sum(x[1:]) / (len(x) - 1))
    f2 = g * (1.0 - np.sqrt(f1 / g))
    return np.array([f1, f2]), g

def CTP1(x):
    x = np.asarray(x, dtype=float)
    objs, g = _ctp_base(x)
    f1, f2 = objs
    g1 = f2 - 0.858 + 0.541 * f1
    g2 = f2 - 0.728 + 0.295 * f1
    return objs, [float(g1), float(g2)]

def CTP2(x):
    x = np.asarray(x, dtype=float)
    objs, g = _ctp_base(x)
    f1, f2 = objs
    theta, a, b, d, e = -0.2 * np.pi, 0.2, 10.0, 1.0, 6.0
    val = np.cos(e * (f2 - f1))
    g_val = (f2 - f1) * np.sin(theta) + (f2 + f1) * np.cos(theta) - a * np.abs(val)**b - d
    return objs, [float(-g_val)]

def CTP3(x):
    x = np.asarray(x, dtype=float)
    objs, g = _ctp_base(x)
    f1, f2 = objs
    theta, a, b, d, e = -0.2 * np.pi, 0.1, 10.0, 1.0, 0.5
    val = np.cos(e * (f2 - f1))
    g_val = (f2 - f1) * np.sin(theta) + (f2 + f1) * np.cos(theta) - a * np.abs(val)**b - d
    return objs, [float(-g_val)]

def CTP4(x):
    x = np.asarray(x, dtype=float)
    objs, g = _ctp_base(x)
    f1, f2 = objs
    theta, a, b, d, e = -0.2 * np.pi, 0.75, 0.5, 0.75, 0.5
    val = np.cos(e * (f2 - f1))
    g_val = (f2 - f1) * np.sin(theta) + (f2 + f1) * np.cos(theta) - a * np.abs(val)**b - d
    return objs, [float(-g_val)]

def CTP5(x):
    x = np.asarray(x, dtype=float)
    objs, g = _ctp_base(x)
    f1, f2 = objs
    theta, a, b, d, e = -0.2 * np.pi, 0.1, 0.5, 0.75, 2.0
    val = np.cos(e * (f2 - f1))
    g_val = (f2 - f1) * np.sin(theta) + (f2 + f1) * np.cos(theta) - a * np.abs(val)**b - d
    return objs, [float(-g_val)]

def CTP6(x):
    x = np.asarray(x, dtype=float)
    objs, g = _ctp_base(x)
    f1, f2 = objs
    theta, a, b, d, e = -0.2 * np.pi, 40.0, 0.5, 1.0, 6.0
    val = np.cos(e * (f2 - f1))
    g_val = (f2 - f1) * np.sin(theta) + (f2 + f1) * np.cos(theta) - a * np.abs(val)**b - d
    return objs, [float(-g_val)]

def CTP7(x):
    x = np.asarray(x, dtype=float)
    objs, g = _ctp_base(x)
    f1, f2 = objs
    theta, a, b, d, e = -0.2 * np.pi, 20.0, 2.0, 1.0, 6.0
    val = np.cos(e * (f2 - f1))
    g_val = (f2 - f1) * np.sin(theta) + (f2 + f1) * np.cos(theta) - a * np.abs(val)**b - d
    return objs, [float(-g_val)]

def CTP8(x):
    x = np.asarray(x, dtype=float)
    objs, g = _ctp_base(x)
    f1, f2 = objs
    g1 = (f2 - f1) * np.sin(-0.2 * np.pi) + (f2 + f1) * np.cos(-0.2 * np.pi) - 0.1
    g2 = (f2 - f1) * np.sin(0.1 * np.pi) + (f2 + f1) * np.cos(0.1 * np.pi) - 0.1
    return objs, [float(-g1), float(-g2)]