from abc import ABC, abstractmethod

import numpy as np


class Function(ABC):
    def __init__(self, vmin=1, vmax=3):
        self.vmin = vmin
        self.vmax = vmax

    @property
    @abstractmethod
    def name(self):
        pass

    @property
    @abstractmethod
    def marker(self):
        pass

    @property
    @abstractmethod
    def fmin(self):
        pass

    @property
    @abstractmethod
    def fmax(self):
        pass

    @abstractmethod
    def value(self, **kwargs):
        pass

    @abstractmethod
    def sequence_to_values(self, seq):
        pass

    def transform(self, value):
        value = (value - self.fmin)/(self.fmax - self.fmin)
        value = value*(self.vmax - self.vmin) + self.vmin
        return value

    def reverse(self, value):
        return self.vmin + self.vmax - value



class Function1D(Function):
    @property
    def fmin(self):
        return min(self.raw_func(x) for x in [self.vmin, self.vmax, (self.vmin+self.vmax)/2])

    @property
    def fmax(self):
        return max(self.raw_func(x) for x in [self.vmin, self.vmax, (self.vmin+self.vmax)/2])

    @abstractmethod
    def raw_func(self, x):
        pass

    def value(self, **kwargs):
        x = kwargs['xn']
        if x is None:
            return np.nan

        return self.transform(self.raw_func(x))

    def sequence_to_values(self, seq):
        return [self.value(xn=seq[i]) for i in range(len(seq))]


class Function2D(Function):
    @property
    def fmin(self):
        return min(self.raw_func(x1, x2) for x1 in [self.vmin, self.vmax] for x2 in [self.vmin, self.vmax])

    @property
    def fmax(self):
        return max(self.raw_func(x1, x2) for x1 in [self.vmin, self.vmax] for x2 in [self.vmin, self.vmax])

    @abstractmethod
    def raw_func(self, x1, x2):
        pass

    def value(self, **kwargs):
        x1, x2 = kwargs['xn'], kwargs['xnm1']
        if x2 is None:
            return np.nan

        return self.transform(self.raw_func(x1, x2))

    def sequence_to_values(self, seq):
        return [np.nan] + [self.value(xn=seq[i + 1], xnm1=seq[i]) for i in range(len(seq) - 1)]


class FunctionReverse(Function):
    def transform(self, value):
        return self.reverse(super().transform(value))

class FunctionMinus(Function):
    def transform(self, value):
        return -super().transform(value)

class Function1DRotate(Function1D):
    def value(self, **kwargs):
        x = kwargs['xn']
        if x is None:
            return np.nan

        return self.transform(self.raw_func(self.vmin + self.vmax - x))


class X(Function1D):
    @property
    def name(self):
        return 'x'

    @property
    def marker(self):
        return 'D'

    @property
    def color(self):
        return 'cornflowerblue'

    def raw_func(self, x):
        return x

class XReverse(X, FunctionReverse):
    @property
    def name(self):
        return 'x_rev'

    @property
    def color(self):
        return 'mediumslateblue'

class X2(Function1D):
    @property
    def name(self):
        return 'x2'

    @property
    def color(self):
        return 'royalblue'

    @property
    def marker(self):
        return 'o'

    def raw_func(self, x):
        return ((x - self.vmin)/self.vmax)**2

class X2Reverse(X2, FunctionReverse):
    @property
    def name(self):
        return 'x2_rev'


class X2Rotate(X2, Function1DRotate):
    @property
    def name(self):
        return 'x2_rot'

    # @property
    # def marker(self):
    #     return 'D'

    @property
    def color(self):
        return 'slateblue'

class X2RotateReverse(X2, Function1DRotate, FunctionReverse):
    @property
    def name(self):
        return 'x2_rot_rev'

    @property
    def marker(self):
        return 'o'

    @property
    def color(self):
        return 'slateblue'

class X3(Function1D):
    @property
    def name(self):
        return 'x3'

    @property
    def marker(self):
        return 'p'

    def raw_func(self, x):
        return ((x - self.vmin)/self.vmax)**3

    @property
    def color(self):
        return 'mediumblue'

class X3Rotate(X3, Function1DRotate):
    @property
    def name(self):
        return 'x3_rot'

    @property
    def color(self):
        return 'darkslateblue'

class X3RotateReverse(X3, Function1DRotate, FunctionReverse):
    @property
    def name(self):
        return 'x3_rot_rev'


class X4(Function1D):
    @property
    def name(self):
        return 'x4'

    @property
    def marker(self):
        return 'v'

    def raw_func(self, x):
        return ((x - self.vmin)/self.vmax)**4

    @property
    def color(self):
        return 'pink'

class X4Reverse(X4, FunctionReverse):
    @property
    def name(self):
        return 'x4_rev'

class X4Rotate(X4, Function1DRotate):
    @property
    def name(self):
        return 'x4_rot'

    @property
    def color(self):
        return 'magenta'

class X4RotateReverse(X4, Function1DRotate, FunctionReverse):
    @property
    def name(self):
        return 'x4_rot_rev'

class X5(Function1D):
    @property
    def name(self):
        return 'x5'

    @property
    def marker(self):
        return 'p'

    def raw_func(self, x):
        return ((x - self.vmin)/self.vmax)**5

class X5Reverse(X5, FunctionReverse):
    @property
    def name(self):
        return 'x5_rev'

class X5Rotate(X5, Function1DRotate):
    @property
    def name(self):
        return 'x5_rot'

class X5RotateReverse(X5, Function1DRotate, FunctionReverse):
    @property
    def name(self):
        return 'x5_rot_rev'

class X4Minus(X4, FunctionMinus):
    @property
    def name(self):
        return 'x4_minus'

class X4RotateMinus(X4, Function1DRotate, FunctionMinus):
    @property
    def name(self):
        return 'x4_rot_minus'


class X6(Function1D):
    @property
    def name(self):
        return 'x6'

    @property
    def marker(self):
        return 'p'

    def raw_func(self, x):
        return ((x - self.vmin)/self.vmax)**6


class X6Reverse(X6, FunctionReverse):
    @property
    def name(self):
        return 'x6_rev'

class X6Rotate(X6, Function1DRotate):
    @property
    def name(self):
        return 'x6_rot'

class X6RotateReverse(X6, Function1DRotate, FunctionReverse):
    @property
    def name(self):
        return 'x6_rot_rev'


class Triangle(Function1D):
    @property
    def name(self):
        return 'triangle'

    @property
    def marker(self):
        return 'v'

    def raw_func(self, x):
        if x <= (self.vmin + self.vmax)/2:
            return 2 * (x - self.vmin)
        else:
            return 2 * (self.vmax - self.vmin) - 2 * (x - self.vmin)


class Poly2(Function1D):
    @property
    def name(self):
        return 'poly2'

    @property
    def marker(self):
        return 'v'

    def raw_func(self, x):
        return (x-3)**2 + 1


class Poly2Reverse(Poly2, FunctionReverse):
    @property
    def name(self):
        return 'poly2_rev'


class Step1D(Function1D):
    @property
    def name(self):
        return 'step'

    @property
    def marker(self):
        return 'p'

    def raw_func(self, x):
        if x < (self.vmin + self.vmax)/2:
            return 1
        else:
            return 5


class Step1DReverse(Step1D, FunctionReverse):
    @property
    def name(self):
        return 'step_rev'


class Bump1D(Function1D):
    @property
    def name(self):
        return 'bump'

    @property
    def marker(self):
        return 's'

    def raw_func(self, x):
        if (self.vmin + self.vmax) / 4 < x < 3*(self.vmin + self.vmax) / 4:
            return 5
        else:
            return 1


class Bump1DReverse(Step1D, FunctionReverse):
    @property
    def name(self):
        return 'bump_rev'


class Tanh(Function1D):
    @property
    def name(self):
        return 'tanh'

    @property
    def marker(self):
        return 's'

    def raw_func(self, x):
        return np.tanh(4*(x-2))

class Tan(Function1D):
    @property
    def name(self):
        return 'tan'

    @property
    def marker(self):
        return 's'

    def raw_func(self, x):
        return np.tan(1.*(x-2)**2*np.sign(x-2))

class Sine(Function1D):
    @property
    def name(self):
        return 'sine'

    @property
    def marker(self):
        return 's'

    def raw_func(self, x):
        return 2*np.sin(0.25*np.pi*(x+1)) + 3

class SineReverse(Sine, FunctionReverse):
    @property
    def name(self):
        return 'sine_rev'


class CoSine(Function1D):
    @property
    def name(self):
        return 'cosine'

    @property
    def marker(self):
        return 'p'

    def raw_func(self, x):
        return 2*np.cos(0.25*np.pi*(x+1)) + 3

class CoSineReverse(Sine, FunctionReverse):
    @property
    def name(self):
        return 'cosine_rev'

class Function2D180(Function2D):
    def value(self, **kwargs):
        x1, x2 = kwargs['xn'], kwargs['xnm1']
        if x2 is None:
            return np.nan

        return self.transform(self.raw_func(self.vmax + self.vmin - x1, self.vmax + self.vmin - x2))


class Function2D90(Function2D):
    def value(self, **kwargs):
        x1, x2 = kwargs['xn'], kwargs['xnm1']
        if x2 is None:
            return np.nan

        return self.transform(self.raw_func(self.vmax + self.vmin - x1, x2))


class Function2D270(Function2D):
    def value(self, **kwargs):
        x1, x2 = kwargs['xn'], kwargs['xnm1']
        if x2 is None:
            return np.nan

        return self.transform(self.raw_func(x1, self.vmax + self.vmin - x2))


class L1(Function2D):
    @property
    def name(self):
        return 'l1'

    @property
    def marker(self):
        return 'D'

    @property
    def color(self):
        return 'cornflowerblue'

    def raw_func(self, x1, x2):
        return abs(x1) + abs(x2)


class L1Reverse(L1, Function2D180):
    @property
    def name(self):
        return 'l1_rev'

    @property
    def color(self):
        return 'mediumslateblue'

class L190(L1, Function2D90):
    @property
    def name(self):
        return 'l1_rot'

class L1270(L1, Function2D270):
    @property
    def name(self):
        return 'l1_rot270'

class L2(Function2D):
    @property
    def name(self):
        return 'l2'

    @property
    def marker(self):
        return 'o'

    @property
    def color(self):
        return 'royalblue'

    def raw_func(self, x1, x2):
        return (x1**2 + x2**2) ** 0.5

class L2Reverse(L2, FunctionReverse):
    @property
    def name(self):
        return 'l2_rev'

class L2180(L2, Function2D180):
    @property
    def name(self):
        return 'l2_rot180'

    @property
    def color(self):
        return 'slateblue'

    @property
    def marker(self):
        return 'o'

class L2180Reverse(L2, Function2D180, FunctionReverse):
    @property
    def name(self):
        return 'l2_rot180_rev'

    @property
    def color(self):
        return 'slateblue'

class L290(L2, Function2D90):
    @property
    def name(self):
        return 'l2_rot90'

    @property
    def marker(self):
        return 'D'

    @property
    def color(self):
        return 'cyan'

class L2270(L2, Function2D270):
    @property
    def name(self):
        return 'l2_rot270'

    @property
    def marker(self):
        return '.'

    @property
    def color(self):
        return 'magenta'

class L3(Function2D):
    @property
    def name(self):
        return 'l3'

    @property
    def color(self):
        return 'mediumblue'

    @property
    def marker(self):
        return 'p'

    def raw_func(self, x1, x2):
        return (x1**3 + x2**3) ** (1/3)

class L3180(L3, Function2D180):
    @property
    def name(self):
        return 'l3_rot180'

    @property
    def color(self):
        return 'darkslateblue'

class L3180Reverse(L3, Function2D180, FunctionReverse):
    @property
    def name(self):
        return 'l3_rot180_rev'


class L4(Function2D):
    @property
    def name(self):
        return 'l4'

    @property
    def marker(self):
        return 'v'

    def raw_func(self, x1, x2):
        return (x1**4 + x2**4) ** 0.25

    @property
    def color(self):
        return 'pink'

class L4Reverse(L4, FunctionReverse):
    @property
    def name(self):
        return 'l4_rev'


class L4180(L4, Function2D180):
    @property
    def name(self):
        return 'l4_rot180'

    @property
    def color(self):
        return 'magenta'

class L4180Reverse(L4, Function2D180, FunctionReverse):
    @property
    def name(self):
        return 'l4_rot180_rev'

class L490(L4, Function2D90):
    @property
    def name(self):
        return 'l4_rot90'

    @property
    def color(self):
        return 'green'

class L4270(L4, Function2D270):
    @property
    def name(self):
        return 'l4_rot270'

    @property
    def color(self):
        return 'darkorange'


class L6(Function2D):
    @property
    def name(self):
        return 'l6'

    @property
    def marker(self):
        return 'v'

    def raw_func(self, x1, x2):
        return (x1**6 + x2**6) ** (1/6)

class L6Reverse(L6, FunctionReverse):
    @property
    def name(self):
        return 'l6_rev'


class L6180(L6, Function2D180):
    @property
    def name(self):
        return 'l6_rot180'

class L6180Reverse(L6, Function2D180, FunctionReverse):
    @property
    def name(self):
        return 'l6_rot180_rev'

class L690(L4, Function2D90):
    @property
    def name(self):
        return 'l6_rot'

class L6270(L4, Function2D270):
    @property
    def name(self):
        return 'l6_rot270'


class LMAX(Function2D):
    @property
    def name(self):
        return 'max'

    @property
    def marker(self):
        return 's'

    def raw_func(self, x1, x2):
        return max(x1, x2)


class LMAXReverse(LMAX, FunctionReverse):
    @property
    def name(self):
        return 'max_rev'


class LMAX180(LMAX, Function2D180):
    @property
    def name(self):
        return 'max_rot180'

class LMAX90(LMAX, Function2D90):
    @property
    def name(self):
        return 'max_rot'

class LMAX1270(LMAX, Function2D270):
    @property
    def name(self):
        return 'max_rot270'


class Square(Function):
    @property
    def name(self):
        return 'square'

    @property
    def marker(self):
        return 'p'

    @property
    def fmin(self):
        return 1

    @property
    def fmax(self):
        return 5

    def value(self, **kwargs):
        x1, x2 = kwargs['xn'], kwargs['xnm1']
        if x2 is None:
            return np.nan
        mid = (self.fmin + self.fmax)/2
        value = 2*max(abs(x1 - mid), abs(x2 - mid)) + 1
        return self.transform(value)

class SquareReverse(Function):
    @property
    def name(self):
        return 'square_rev'

    @property
    def marker(self):
        return 'p'

    @property
    def fmin(self):
        return 1

    @property
    def fmax(self):
        return 5

    def value(self, **kwargs):
        x1, x2 = kwargs['xn'], kwargs['xnm1']
        if x2 is None:
            return np.nan
        mid = (self.fmin + self.fmax)/2
        value = 2*max(abs(x1 - mid), abs(x2 - mid)) + 1
        return self.reverse(self.transform(value))


class Identity(Function):
    @property
    def name(self):
        return 'identity'

    @property
    def marker(self):
        return 'p'

    @property
    def fmin(self):
        return 1

    @property
    def fmax(self):
        return 5

    def value(self, **kwargs):
        x1, x2 = kwargs['xn'], kwargs['xnm1']
        if x2 is None:
            return np.nan
        return x1

class IntervalFunction(Function):
    # @property
    # def name(self):
    #     return 'interval'

    @property
    def marker(self):
        return 'p'

    @property
    def fmin(self):
        return min(self.raw_func(15), self.raw_func(44))

    @property
    def fmax(self):
        return max(self.raw_func(15), self.raw_func(44))

    @abstractmethod
    def raw_func(self, interval):
        pass

    def value(self, **kwargs):
        x1, x2 = kwargs['xn'], kwargs['xnm1']
        if x2 is None:
            return np.nan
        interval = kwargs['interval']
        return self.transform(self.raw_func(interval))

    def sequence_to_values(self, seq):
        return [np.nan] + [self.value(xn=0, xnm1=0, interval=seq[j+1] -seq[j]) for j in range(len(seq) - 2)]


class IntervalX(IntervalFunction):
    @property
    def name(self):
        return 'int_x'

    def raw_func(self, interval):
        return interval

    @property
    def marker(self):
        return 'D'


class IntervalXReverse(IntervalX, FunctionReverse):
    @property
    def name(self):
        return 'int_x_rev'


class FunctionIntervalRotate(IntervalFunction):
    def value(self, **kwargs):
        x1, x2 = kwargs['xn'], kwargs['xnm1']
        if x2 is None:
            return np.nan
        interval = kwargs['interval']
        return self.transform(self.raw_func(15 + 44 - interval))


class IntervalX2(IntervalFunction):
    @property
    def name(self):
        return 'int_x2'

    def raw_func(self, interval):
        return interval**2

    @property
    def marker(self):
        return 'o'


class IntervalX2Rotate(IntervalX2, FunctionIntervalRotate):
    @property
    def name(self):
        return 'int_x2_rot'

class IntervalX2Reverse(IntervalX2, FunctionReverse):
    @property
    def name(self):
        return 'int_x2_rev'


class IntervalX2RotateReverse(IntervalX2Rotate, FunctionReverse):
    @property
    def name(self):
        return 'int_x2_rot_rev'


class IntervalX4(IntervalFunction):
    @property
    def name(self):
        return 'int_x4'

    @property
    def marker(self):
        return 'v'

    def raw_func(self, interval):
        return interval ** 4


class IntervalX4Rotate(IntervalX4, FunctionIntervalRotate):
    @property
    def name(self):
        return 'int_x4_rot'


class IntervalX4Reverse(IntervalX4, FunctionReverse):
    @property
    def name(self):
        return 'int_x4_rev'


class IntervalX4RotateReverse(IntervalX4Rotate, FunctionReverse):
    @property
    def name(self):
        return 'int_x4_rot_rev'