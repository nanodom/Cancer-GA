import numpy as np

def CF1(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    j1 = np.arange(2, n, 2)
    j2 = np.arange(1, n, 2)
    
    y = x - np.sin(6 * np.pi * x[0] + np.arange(1, n) * np.pi / n)
    
    f1 = x[0] + (2.0 / len(j1)) * np.sum(y[j1]**2)
    f2 = 1.0 - x[0] + (2.0 / len(j2)) * np.sum(y[j2]**2)
    
    g1 = f2 + f1**2 - 1.0
    return np.array([f1, f2]), [float(-g1)]

def CF2(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    j1 = np.arange(2, n, 2)
    j2 = np.arange(1, n, 2)
    
    y = x.copy()
    y[1:] = x[1:] - np.sin(6 * np.pi * x[0] + np.arange(2, n + 1) * np.pi / n)
    
    f1 = x[0] + (2.0 / len(j1)) * np.sum(y[j1]**2)
    f2 = 1.0 - np.sqrt(x[0]) + (2.0 / len(j2)) * np.sum(y[j2]**2)
    
    t = f2 + np.sqrt(f1) - np.sin(2 * np.pi * (f2 - np.sqrt(f1) + 1.0)) - 1.0
    return np.array([f1, f2]), [float(-t)]

def CF3(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    j1 = np.arange(2, n, 2)
    j2 = np.arange(1, n, 2)
    
    y = x.copy()
    y[1:] = x[1:] - x[0]**(0.5 * (1.0 + 3.0 * (np.arange(2, n + 1) - 2.0) / (n - 2.0)))
    
    f1 = x[0] + (2.0 / len(j1)) * (4.0 * np.sum(y[j1]**2) - 2.0 * np.prod(np.cos(20.0 * y[j1] * np.pi / np.sqrt(j1 + 1))) + 2.0)
    f2 = 1.0 - x[0]**2 + (2.0 / len(j2)) * (4.0 * np.sum(y[j2]**2) - 2.0 * np.prod(np.cos(20.0 * y[j2] * np.pi / np.sqrt(j2 + 1))) + 2.0)
    
    g1 = f2 + f1**2 - 1.0
    return np.array([f1, f2]), [float(-g1)]

def CF4(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    j1 = np.arange(2, n, 2)
    j2 = np.arange(1, n, 2)
    
    y = x.copy()
    y[1:] = x[1:] - np.sin(6 * np.pi * x[0] + np.arange(2, n + 1) * np.pi / n)
    
    f1 = x[0] + (2.0 / len(j1)) * np.sum(y[j1]**2)
    f2 = 1.0 - x[0]**2 + (2.0 / len(j2)) * np.sum(y[j2]**2)
    
    t = f2 + f1**2 - np.sin(2 * np.pi * (f2 - f1**2 + 1.0)) - 1.0
    return np.array([f1, f2]), [float(-t)]

def CF5(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    j1 = np.arange(2, n, 2)
    j2 = np.arange(1, n, 2)
    
    y = x.copy()
    y[1:] = x[1:] - np.sin(6 * np.pi * x[0] + np.arange(2, n + 1) * np.pi / n)
    
    f1 = x[0] + (2.0 / len(j1)) * np.sum(2.0 * y[j1]**2 - np.cos(4 * np.pi * y[j1]) + 1.0)
    f2 = 1.0 - x[0] + (2.0 / len(j2)) * np.sum(2.0 * y[j2]**2 - np.cos(4 * np.pi * y[j2]) + 1.0)
    
    g1 = f2 + f1 - 1.0
    return np.array([f1, f2]), [float(-g1)]

def CF6(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    j1 = np.arange(2, n, 2)
    j2 = np.arange(1, n, 2)
    
    y = x.copy()
    y[1:] = x[1:] - np.sin(6 * np.pi * x[0] + np.arange(2, n + 1) * np.pi / n)
    
    f1 = x[0] + (2.0 / len(j1)) * np.sum(y[j1]**2)
    f2 = (1.0 - x[0])**2 + (2.0 / len(j2)) * np.sum(y[j2]**2)
    
    g1 = f2 + f1**2 - 1.0
    return np.array([f1, f2]), [float(-g1)]

def CF7(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    j1 = np.arange(2, n, 2)
    j2 = np.arange(1, n, 2)
    
    y = x.copy()
    y[1:] = x[1:] - np.sin(6 * np.pi * x[0] + np.arange(2, n + 1) * np.pi / n)
    
    f1 = x[0] + (2.0 / len(j1)) * np.sum(y[j1]**2)
    f2 = (1.0 - x[0])**2 + (2.0 / len(j2)) * np.sum(y[j2]**2)
    
    g1 = f2 + f1**2 - 1.0
    return np.array([f1, f2]), [float(-g1)]

def CF8(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    j1 = np.arange(2, n, 3)
    j2 = np.arange(3, n, 3)
    j3 = np.arange(1, n, 3)
    
    y = x.copy()
    y[2:] = x[2:] - 2 * x[1] * np.sin(2 * np.pi * x[0] + np.arange(3, n + 1) * np.pi / n)
    
    f1 = np.cos(0.5 * np.pi * x[0]) * np.cos(0.5 * np.pi * x[1]) + (2.0 / len(j1)) * np.sum(y[j1]**2)
    f2 = np.cos(0.5 * np.pi * x[0]) * np.sin(0.5 * np.pi * x[1]) + (2.0 / len(j2)) * np.sum(y[j2]**2)
    f3 = np.sin(0.5 * np.pi * x[0]) + (2.0 / len(j3)) * np.sum(y[j3]**2)
    
    g1 = (f1**2 + f2**2) / (1.0 - f3**2) - 1.0
    return np.array([f1, f2, f3]), [float(-g1)]

def CF9(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    j1 = np.arange(2, n, 3)
    j2 = np.arange(3, n, 3)
    j3 = np.arange(1, n, 3)
    
    y = x.copy()
    y[2:] = x[2:] - 2 * x[1] * np.sin(2 * np.pi * x[0] + np.arange(3, n + 1) * np.pi / n)
    
    f1 = np.cos(0.5 * np.pi * x[0]) * np.cos(0.5 * np.pi * x[1]) + (2.0 / len(j1)) * np.sum(y[j1]**2)
    f2 = np.cos(0.5 * np.pi * x[0]) * np.sin(0.5 * np.pi * x[1]) + (2.0 / len(j2)) * np.sum(y[j2]**2)
    f3 = np.sin(0.5 * np.pi * x[0]) + (2.0 / len(j3)) * np.sum(y[j3]**2)
    
    g1 = (f1**2 + f2**2) / (1.0 - f3**2) - 1.0
    return np.array([f1, f2, f3]), [float(-g1)]

def CF10(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    j1 = np.arange(2, n, 3)
    j2 = np.arange(3, n, 3)
    j3 = np.arange(1, n, 3)
    
    y = x.copy()
    y[2:] = x[2:] - 2 * x[1] * np.sin(2 * np.pi * x[0] + np.arange(3, n + 1) * np.pi / n)
    
    f1 = np.cos(0.5 * np.pi * x[0]) * np.cos(0.5 * np.pi * x[1]) + (2.0 / len(j1)) * np.sum(2.0 * y[j1]**2 - np.cos(4 * np.pi * y[j1]) + 1.0)
    f2 = np.cos(0.5 * np.pi * x[0]) * np.sin(0.5 * np.pi * x[1]) + (2.0 / len(j2)) * np.sum(2.0 * y[j2]**2 - np.cos(4 * np.pi * y[j2]) + 1.0)
    f3 = np.sin(0.5 * np.pi * x[0]) + (2.0 / len(j3)) * np.sum(2.0 * y[j3]**2 - np.cos(4 * np.pi * y[j3]) + 1.0)
    
    g1 = (f1**2 + f2**2) / (1.0 - f3**2) - 1.0
    return np.array([f1, f2, f3]), [float(-g1)]