# -*- coding: utf-8 -*-
"""
SuSTech OPAL Lab

A test function sets for optimization from Yi-jun Yang (Total 38 functions)

All functions are scalable and with closed form for fast evaluation

Using example:"

import testfuns as tfs
f = tfs.Ackley(dim) # you need to pre-define the dimension of function i.e. "dim". Then
you can use "f.do_evaluate" to compute a single-value result according solution x.
x: a ndarray with size [dim]. Of course, you can get other attributes by f.bounds, etc."

"""
import math
import numpy as np
np.seterr(all = 'ignore')

def lzip(*args):
    """
    returns zipped result as a list.
    """
    return list(zip(*args))

class Ackley(object):
    def __init__(self, dim):
#        super(Ackley, self).__init__(dim)
        self.dim = dim
        self.bounds = np.array(lzip([-10] * self.dim, [30] * self.dim))
        self.min_loc = [0] * self.dim
        self.fmin = 0
        self.fmax = 22.26946404462
        self.classifiers = ['complicated', 'oscillatory', 'unimodal', 'noisy']

    def do_evaluate(self, x):
        a = 20
        b = 0.2
        c = 2 * math.pi
        return (-a * math.exp(-b * np.sqrt(1.0 / self.dim * sum(x ** 2))) -
                math.exp(1.0 / self.dim * sum(np.cos(c * x))) + a + math.exp(1))
        
class Alpine01(object):
    def __init__(self, dim):
#        super(Alpine01, self).__init__(dim)
        self.dim = dim
        self.bounds = np.array(lzip([-6] * self.dim, [10] * self.dim))
        self.min_loc = [0] * self.dim
        self.fmin = 0
        self.fmax = 8.71520568065 * self.dim
        self.classifiers = ['nonsmooth']

    def do_evaluate(self, x):
        return sum(abs(x * np.sin(x) + 0.1 * x))
    
class ArithmeticGeometricMean(object):
    def __init__(self, dim):
#        super(ArithmeticGeometricMean, self).__init__(dim)
        self.dim = dim
        self.bounds = np.array(lzip([0] * self.dim, [10] * self.dim))
        self.min_loc = [0] * self.dim
        self.fmin = 0
        self.fmax = (10 * (self.dim - 1.0) / self.dim) ** 2
        self.classifiers = ['bound_min', 'boring', 'multi_min']

    def do_evaluate(self, x):
        return (np.mean(x) - np.prod(x) ** (1.0 / self.dim)) ** 2
    
class Cigar(object):
    def __init__(self, dim):
#        assert dim > 1
#        super(Cigar, self).__init__(dim)
        self.dim = dim
        self.bounds = np.array(lzip([-1] * self.dim, [1] * self.dim))
        self.min_loc = [0] * self.dim
        self.fmin = 0
        self.fmax = 1 + 1e6 * self.dim
        self.classifiers = ['unimodal', 'unscaled']

    def do_evaluate(self, x):
        return x[0] ** 2 + 1e6 * sum(x[1:] ** 2)
    
class CosineMixture(object):
    def __init__(self, dim):
#        super(CosineMixture, self).__init__(dim)
        self.dim = dim
        self.bounds = np.array(lzip([-1] * self.dim, [1] * self.dim))
        self.min_loc = [0.184872823182918] * self.dim
        self.fmin = -0.063012202176250 * self.dim
        self.fmax = 0.9 * self.dim
        self.classifiers = ['oscillatory', 'multi_min']

    def do_evaluate(self, x):
        return 0.1 * sum(np.cos(5 * math.pi * x)) + sum(x ** 2)
    
class Csendes(object):
    def __init__(self, dim):
#        super(Csendes, self).__init__(dim)
        self.dim = dim
        self.bounds = np.array(lzip([-0.5] * self.dim, [1] * self.dim))
        self.min_loc = [0] * self.dim
        self.fmin = 0
        self.fmax = self.do_evaluate(np.asarray([1] * self.dim))
        self.classifiers = ['unimodal']

    def do_evaluate(self, x):
        return sum((x ** 6) * (2 + np.sin(1 / (x + np.finfo(float).eps))))
    
class Deb01(object):
    def __init__(self, dim):
#        super(Deb01, self).__init__(dim)
        self.dim = dim
        self.bounds = np.array(lzip([-1] * self.dim, [1] * self.dim))
        self.min_loc = [0.3] * self.dim
        self.fmin = -1
        self.fmax = 0
        self.classifiers = ['oscillatory', 'multi_min']

    def do_evaluate(self, x):
        return -(1.0 / self.dim) * sum(np.sin(5 * math.pi * x) ** 6)
    
class Deb02(object):
    def __init__(self, dim):
#        super(Deb02, self).__init__(dim)
        self.dim = dim
        self.bounds = np.array(lzip([0] * self.dim, [1] * self.dim))
        self.min_loc = [0.0796993926887] * self.dim
        self.fmin = -1
        self.fmax = 0
        self.classifiers = ['oscillatory', 'multi_min']

    def do_evaluate(self, x):
        return -(1.0 / self.dim) * sum(np.sin(5 * math.pi * (x ** 0.75 - 0.05)) ** 6)
    
class DeflectedCorrugatedSpring(object):
    def __init__(self, dim):
#        super(DeflectedCorrugatedSpring, self).__init__(dim)
        self.dim = dim
        self.alpha = 5.0
        self.K = 5.0
        self.bounds = np.array(lzip([0] * self.dim, [1.5 * self.alpha] * self.dim))
        self.min_loc = [self.alpha] * self.dim
        self.fmin = self.do_evaluate(np.asarray(self.min_loc))
        self.fmax = self.do_evaluate(np.zeros(self.dim))
        self.classifiers = ['oscillatory']

    def do_evaluate(self, x):
        return -np.cos(self.K * np.sqrt(sum((x - self.alpha) ** 2))) + 0.1 * sum((x - self.alpha) ** 2)
    
class DropWave(object):
    def __init__(self, dim):
#        super(DropWave, self).__init__(dim)
        self.dim = dim
        self.bounds = np.array(lzip([-2] * self.dim, [5.12] * self.dim))
        self.min_loc = [0] * self.dim
        self.fmin = -1
        self.fmax = 0
        self.classifiers = ['oscillatory']

    def do_evaluate(self, x):
        norm_x = sum(x ** 2)
        return -(1 + np.cos(12 * np.sqrt(norm_x))) / (0.5 * norm_x + 2)
    
class Easom(object):
    def __init__(self, dim):
#        super(Easom, self).__init__(dim)
        self.dim = dim
        self.bounds = np.array(lzip([-100] * self.dim, [20] * self.dim))
        self.min_loc = [0] * self.dim
        self.fmin = 0
        self.fmax = 22.3504010789
        self.classifiers = ['unimodal', 'boring']

    def do_evaluate(self, x):
        a = 20
        b = 0.2
        c = 2 * math.pi
        n = self.dim
        return -a * np.exp(-b * np.sqrt(sum(x ** 2) / n)) - np.exp(sum(np.cos(c * x)) / n) + a + np.exp(1)
    
class Exponential(object):
    def __init__(self, dim):
#        super(Exponential, self).__init__(dim)
        self.dim = dim
        self.bounds = np.array(lzip([-0.7] * self.dim, [0.2] * self.dim))
        self.min_loc = [0] * self.dim
        self.fmin = -1
        self.fmax = self.do_evaluate(np.asarray([-0.7] * self.dim))
        self.classifiers = ['unimodal']

    def do_evaluate(self, x):
        return -np.exp(-0.5 * sum(x ** 2))
    
class ManifoldMin(object):
    def __init__(self, dim):
#        super(ManifoldMin, self).__init__(dim)
        self.dim = dim
        self.bounds = np.array(lzip([-10] * self.dim, [10] * self.dim))
        self.min_loc = [0] * self.dim
        self.fmin = 0
        self.fmax = self.do_evaluate([10] * self.dim)
        self.classifiers = ['nonsmooth', 'multi_min', 'unscaled']

    def do_evaluate(self, x):
        return sum(np.abs(x)) * np.prod(np.abs(x))
    
class Perm01(object):
    def __init__(self, dim):
#        assert dim > 1
#        super(Perm01, self).__init__(dim)
        self.dim = dim
        self.bounds = np.array(lzip([-self.dim] * self.dim, [self.dim + 1] * self.dim))
        self.min_loc = [1] * self.dim
        self.fmin = 0
        self.fmax = self.do_evaluate([self.dim + 1] * self.dim)
        self.classifiers = ['unscaled']

    def do_evaluate(self, x):
        return sum(
            sum([(j ** k + 0.5) * ((x[j - 1] / j) ** k - 1) for j in range(1, self.dim + 1)]) ** 2
            for k in range(1, self.dim + 1)
        )
        
class Perm02(object):
    def __init__(self, dim):
#        assert dim > 1
#        super(Perm02, self).__init__(dim)
        self.dim = dim
        self.bounds = np.array(lzip([-self.dim] * self.dim, [self.dim + 1] * self.dim))
        self.min_loc = 1 / np.arange(1, self.dim + 1)
        self.fmin = 0
        self.fmax = self.do_evaluate([self.dim + 1] * self.dim)
        self.classifiers = ['unscaled']

    def do_evaluate(self, x):
        return sum(
            sum([(j + 10) * (x[j - 1]**k - (1.0 / j)**k) for j in range(1, self.dim + 1)]) ** 2
            for k in range(1, self.dim + 1)
        )
        
class Pinter(object):
    def __init__(self, dim):
#        assert dim > 1
#        super(Pinter, self).__init__(dim)
        self.dim = dim
        self.bounds = np.array(lzip([-5] * self.dim, [2] * self.dim))
        self.min_loc = [0] * self.dim
        self.fmin = 0
        self.fmax = self.do_evaluate([-5] * self.dim)

    def do_evaluate(self, x):
        f = 0
        for i in range(self.dim):
            x_i = x[i]
            if i == 0:
                x_mi = x[-1]
                x_pi = x[i + 1]
            elif i == self.dim - 1:
                x_mi = x[i - 1]
                x_pi = x[0]
            else:
                x_mi = x[i - 1]
                x_pi = x[i + 1]
            a = x_mi * np.sin(x_i) + np.sin(x_pi)
            b = x_mi ** 2 - 2 * x_i + 3 * x_pi - np.cos(x_i) + 1
            f += (i + 1) * x_i ** 2 + 20 * (i + 1) * np.sin(a) ** 2 + (i + 1) * np.log10(1 + (i + 1) * b ** 2)
        return f
    
class Plateau(object):
    def __init__(self, dim):
#        super(Plateau, self).__init__(dim)
        self.dim = dim
        self.bounds = np.array(lzip([-2.34] * self.dim, [5.12] * self.dim))
        self.min_loc = [0] * self.dim
        self.fmin = 30
        self.fmax = self.do_evaluate([5.12] * self.dim)
        self.classifiers = ['discrete', 'unimodal']

    def do_evaluate(self, x):
        return 30 + sum(np.floor(np.abs(x)))
    
class RippleSmall(object):
    def __init__(self, dim):
#        assert dim == 2
#        super(RippleSmall, self).__init__(dim)
        self.dim = dim
        self.bounds = np.array(lzip([0] * self.dim, [1] * self.dim))
        self.min_loc = [0.1] * self.dim
        self.fmin = -2.2
        self.fmax = 0
        self.classifiers = ['oscillatory']

    def do_evaluate(self, x):
        return sum(-np.exp(-2 * np.log(2) * ((x - 0.1) / 0.8) ** 2) * (np.sin(5 * math.pi * x) ** 6 + 0.1 * np.cos(500 * math.pi * x) ** 2))
    
class RippleBig(object):
    def __init__(self, dim):
#        assert dim == 2
#        super(RippleBig, self).__init__(dim)
        self.dim = dim
        self.bounds = np.array(lzip([0] * self.dim, [1] * self.dim))
        self.min_loc = [0.1] * self.dim
        self.fmin = -2
        self.fmax = 0
        self.classifiers = ['oscillatory']

    def do_evaluate(self, x):
        return sum(-np.exp(-2 * np.log(2) * ((x - 0.1) / 0.8) ** 2) * (np.sin(5 * math.pi * x) ** 6))
    
class RosenbrockLog(object):
    def __init__(self, dim):
#        assert dim == 11
#        super(RosenbrockLog, self).__init__(dim)
        self.dim = dim
        self.bounds = np.array(lzip([-30] * self.dim, [30] * self.dim))
        self.min_loc = [1] * self.dim
        self.fmin = 0
        self.fmax = 10.09400460102

    def do_evaluate(self, x):
        return np.log(1 + sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))
    
class Salomon(object):
    def __init__(self, dim):
#        super(Salomon, self).__init__(dim)
        self.dim = dim
        self.bounds = np.array(lzip([-100] * self.dim, [50] * self.dim))
        self.min_loc = [0] * self.dim
        self.fmin = 0
        self.fmax = self.do_evaluate(np.asarray([-100] * self.dim))
        self.classifiers = ['oscillatory']

    def do_evaluate(self, x):
        return 1 - np.cos(2 * math.pi * np.sqrt(sum(x ** 2))) + 0.1 * np.sqrt(sum(x ** 2))
    
class Sargan(object):
    def __init__(self, dim):
#        assert dim > 1
#        super(Sargan, self).__init__(dim)
        self.dim = dim
        self.bounds = np.array(lzip([-2] * self.dim, [4] * self.dim))
        self.min_loc = [0] * self.dim
        self.fmin = 0
        self.fmax = self.do_evaluate(np.asarray([4] * self.dim))
        self.classifiers = ['unimodal']

    def do_evaluate(self, x):
        x0 = x[:-1]
        x1 = np.roll(x, -1)[:-1]
        return sum(self.dim * (x ** 2 + 0.4 * sum(x0 * x1)))
    
class Schwefel01(object):
    def __init__(self, dim):
#        super(Schwefel01, self).__init__(dim)
        self.dim = dim
        self.bounds = np.array(lzip([-100] * self.dim, [20] * self.dim))
        self.min_loc = [0] * self.dim
        self.fmin = 0
        self.fmax = self.do_evaluate(np.asarray([-100] * self.dim))
        self.classifiers = ['unscaled', 'unimodal']

    def do_evaluate(self, x):
        return (sum(x ** 2)) ** np.sqrt(math.pi)
    
class Schwefel20(object):
    def __init__(self, dim):
#        super(Schwefel20, self).__init__(dim)
        self.dim = dim
        self.bounds = np.array(lzip([-60] * self.dim, [100] * self.dim))
        self.min_loc = [0] * self.dim
        self.fmin = 0
        self.fmax = self.do_evaluate(np.asarray([100] * self.dim))
        self.classifiers = ['unimodal', 'nonsmooth']

    def do_evaluate(self, x):
        return sum(np.abs(x))
    
class Schwefel22(object):
    def __init__(self, dim):
#        super(Schwefel22, self).__init__(dim)
        self.dim = dim
        self.bounds = np.array(lzip([-5] * self.dim, [10] * self.dim))
        self.min_loc = [0] * self.dim
        self.fmin = 0
        self.fmax = self.do_evaluate(np.asarray([10] * self.dim))
        self.classifiers = ['unimodal', 'nonsmooth']

    def do_evaluate(self, x):
        return sum(np.abs(x)) + np.prod(abs(x))
    
class Schwefel26(object):
    def __init__(self, dim):
#        assert dim == 2
#        super(Schwefel26, self).__init__(dim)
        self.dim = dim
        self.bounds = np.array(lzip([-500] * self.dim, [500] * self.dim))
        self.min_loc = [420.968746] * self.dim
        self.fmin = 0
        self.fmax = 1675.92130876
        self.classifiers = ['oscillatory', 'multimin']

    def do_evaluate(self, x):
        return -np.sum(x * np.sin(np.sqrt(np.abs(x)))) + 418.9829 * self.dim
    
class Shubert01(object):
    def __init__(self, dim):
#        assert dim == 2
#        super(Shubert01, self).__init__(dim)
        self.dim = dim
        self.bounds = np.array(lzip([-10] * self.dim, [10] * self.dim))
        self.min_loc = [-7.0835, 4.8580]
        self.fmin = -186.7309
        self.fmax = 210.448484805
        self.classifiers = ['multi_min', 'oscillatory']

    def do_evaluate(self, x):
        return np.prod([sum([i * np.cos((i + 1) * xx + i) for i in range(1, 6)]) for xx in x])
    
class SineEnvelope(object):
    def __init__(self, dim):
#        assert dim > 1
#        super(SineEnvelope, self).__init__(dim)
        self.dim = dim
        self.bounds = np.array(lzip([-20] * self.dim, [10] * self.dim))
        self.min_loc = [0] * self.dim
        self.fmin = 0
        self.fmax = self.dim - 1
        self.classifiers = ['oscillatory']

    def do_evaluate(self, x):
        x_sq = x[0:-1] ** 2 + x[1:] ** 2
        return sum((np.sin(np.sqrt(x_sq)) ** 2 - 0.5) / (1 + 0.001 * x_sq) ** 2 + 0.5)
    
class Step(object):
    def __init__(self, dim):
#        super(Step, self).__init__(dim)
        self.dim = dim
        self.bounds = np.array(lzip([-5] * self.dim, [5] * self.dim))
        self.min_loc = [0.5] * self.dim
        self.fmin = self.do_evaluate(np.asarray([0] * self.dim))
        self.fmax = self.do_evaluate(np.asarray([5] * self.dim))
        self.classifiers = ['discrete', 'unimodal']

    def do_evaluate(self, x):
        return sum((np.floor(x) + 0.5) ** 2)
    
class StretchedV(object):
    def __init__(self, dim):
#        assert dim == 2
#        super(StretchedV, self).__init__(dim)
        self.dim = dim
        self.bounds = np.array(lzip([-10] * self.dim, [5] * self.dim))
        self.min_loc = [-9.38723188, 9.34026753]
        self.fmin = 0
        self.fmax = 3.47171564062
        self.classifiers = ['oscillatory',  'multi_min']

    def do_evaluate(self, x):
        r = sum(x ** 2)
        return r ** 0.25 * (np.sin(50 * r ** 0.1 + 1)) ** 2
    
class StyblinskiTang(object):
    def __init__(self, dim):
#        super(StyblinskiTang, self).__init__(dim)
        self.dim = dim
        self.bounds = np.array(lzip([-5] * self.dim, [5] * self.dim))
        self.min_loc = [-2.903534018185960] * self.dim
        self.fmin = -39.16616570377142 * self.dim
        self.fmax = self.do_evaluate(np.asarray([5] * self.dim))

    def do_evaluate(self, x):
        return sum(x ** 4 - 16 * x ** 2 + 5 * x) / 2
    
class SumPowers(object):
    def __init__(self, dim):
#        super(SumPowers, self).__init__(dim)
        self.dim = dim
        self.bounds = np.array(lzip([-1] * self.dim, [0.5] * self.dim))
        self.min_loc = [0] * self.dim
        self.fmin = 0
        self.fmax = self.do_evaluate(np.asarray([-1] * self.dim))
        self.classifiers = ['unimodal','boring']

    def do_evaluate(self, x):
        return np.sum([np.abs(x) ** (i + 1) for i in range(1, self.dim + 1)])
    
class Trid(object):
    def __init__(self, dim):
#        assert dim == 6
#        super(Trid, self).__init__(dim)
        self.dim = dim
        self.bounds = np.array(lzip([0] * self.dim, [20] * self.dim))
        self.min_loc = [6, 10, 12, 12, 10, 6]
        self.fmin = -50
        self.fmax = 1086
        self.classifiers = ['unimodal','boring']

    def do_evaluate(self, x):
        return sum((x - 1) ** 2) - sum(x[1:] * x[0:-1])
    
class Weierstrass(object):
    def __init__(self, dim):
#        super(Weierstrass, self).__init__(dim)
        self.dim = dim
        self.bounds = np.array(lzip([-0.5] * self.dim, [0.2] * self.dim))
        self.min_loc = [0] * self.dim
        self.fmin = self.do_evaluate(np.asarray(self.min_loc))
        self.fmax = self.do_evaluate(np.asarray([-0.5] * self.dim))
        self.classifiers = ['complicated']

    def do_evaluate(self, x):
        a, b, kmax = 0.5, 3, 20
        ak = a ** (np.arange(0, kmax + 1))
        bk = b ** (np.arange(0, kmax + 1))
        return np.sum([np.sum(ak * np.cos(2 * math.pi * bk * (xx + 0.5))) - self.dim * np.sum(ak * np.cos(math.pi * bk)) for xx in x])
    
class XinSheYang02(object):
    def __init__(self, dim):
#        assert dim == 2
#        super(XinSheYang02, self).__init__(dim)
        self.dim = dim
        self.bounds = np.array(lzip([-math.pi] * self.dim, [2 * math.pi] * self.dim))
        self.min_loc = [0] * self.dim
        self.fmin = 0
        self.fmax = 88.8266046808
        self.classifiers = ['nonsmooth', 'unscaled']

    def do_evaluate(self, x):
        return sum(np.abs(x)) * np.exp(-sum(np.sin(x ** 2)))
    
class XinSheYang03(object):
    def __init__(self, dim):
#        assert dim == 2
#        super(XinSheYang03, self).__init__(dim)
        self.dim = dim
        self.bounds = np.array(lzip([-10] * self.dim, [20] * self.dim))
        self.min_loc = [0] * self.dim
        self.fmin = -1
        self.fmax = 1
        self.classifiers = ['boring', 'unimodal']

    def do_evaluate(self, x):
        beta, m = 15, 5
        return np.exp(-sum((x / beta) ** (2 * m))) - 2 * np.exp(-sum(x ** 2)) * np.prod(np.cos(x) ** 2)
    
class YaoLiu(object):
    def __init__(self, dim):
#        super(YaoLiu, self).__init__(dim)
        self.dim = dim
        self.bounds = np.array(lzip([-5.12] * self.dim, [2] * self.dim))
        self.min_loc = [0] * self.dim
        self.fmin = 0
        self.fmax = self.do_evaluate(np.asarray([-4.52299366685] * self.dim))
        self.classifiers = ['oscillatory','as same as Rastrigin']

    def do_evaluate(self, x):
        return sum(x ** 2 - 10 * np.cos(2 * math.pi * x) + 10)
    
class ZeroSum(object):
    def __init__(self, dim):
#        super(ZeroSum, self).__init__(dim)
        self.dim = dim
        self.bounds = np.array(lzip([-8] * self.dim, [6] * self.dim))
        self.min_loc = [0] * self.dim
        self.fmin = 1
        self.fmax = self.do_evaluate(np.asarray([-8] * self.dim))
        self.classifiers = ['nonsmooth', 'multi_min']

    def do_evaluate(self, x):
        return 1 + (10000 * np.abs(sum(x))) ** 0.5