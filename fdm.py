import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from timeit import default_timer as timer
import tracemalloc


class FDM:
    def __init__(self, maxk, n, L=1, T=2, alpha=0.25):
        self.L = L
        self.T = T
        self.maxk = maxk
        self.n = n
        self.alpha = alpha
        self.dt = T / maxk
        self.dx = L / n
        self.th = alpha * self.dt / self.dx ** 2
        x = np.zeros(n + 1)
        u = np.zeros((n + 1, maxk))

        for i in range(n + 1):
            x[i] = i * self.dx
            u[i, 0] = np.sin(np.pi * x[i])

        self.u = u
        self.sol_mat = self.get_sol_mat()

        self.expl_euler_mat = None
        self.impl_euler_mat = None
        self.trapz_euler_mat = None
        self.rk4_mat = None

    def get_expl_mat(self):
        expl_euler_mat = np.copy(self.u)

        for k in range(1, self.maxk):
            for i in range(1, self.n):
                expl_euler_mat[i, k] = expl_euler_mat[i, k - 1] + \
                                       self.th * (expl_euler_mat[i - 1, k - 1] +
                                                  expl_euler_mat[i + 1, k - 1] -
                                                  2 * expl_euler_mat[i, k - 1])

        self.expl_euler_mat = expl_euler_mat
        return expl_euler_mat

    def expl_euler(self):
        if self.expl_euler_mat is not None:
            return self.expl_euler_mat

        return self.get_expl_mat()

    def get_impl_mat(self):
        impl_euler_mat = np.copy(self.u)

        a_diag = [1 + 2 * self.th] * (self.n - 1)
        b_diag = [-self.th] * (self.n - 2)
        impl_mat = np.linalg.inv(np.diag(a_diag) + np.diag(b_diag, -1) + np.diag(b_diag, 1))

        for k in range(1, self.maxk):
            impl_euler_mat[1:self.n, k] = impl_mat @ impl_euler_mat[1:self.n, k - 1]

        self.impl_euler_mat = impl_euler_mat
        return impl_euler_mat

    def impl_euler(self):
        if self.impl_euler_mat is not None:
            return self.impl_euler_mat

        return self.get_impl_mat()

    def get_trapz_mat(self):
        trapz_euler_mat = np.copy(self.u)

        l_diag = [2 + 2 * self.th] * (self.n - 1)
        r_diag = [2 - 2 * self.th] * (self.n - 1)
        s_diag = [self.th] * (self.n - 2)

        l_mat = np.diag(l_diag) - np.diag(s_diag, -1) - np.diag(s_diag, 1)
        r_mat = np.diag(r_diag) + np.diag(s_diag, -1) + np.diag(s_diag, 1)

        comb_mat = np.linalg.inv(l_mat) @ r_mat

        for k in range(1, self.maxk):
            cur_col = trapz_euler_mat[1:self.n, k - 1]
            trapz_euler_mat[1:self.n, k] = comb_mat @ cur_col

        self.trapz_euler_mat = trapz_euler_mat
        return trapz_euler_mat

    def trapz_euler(self):
        if self.trapz_euler_mat is not None:
            return self.trapz_euler_mat

        return self.get_trapz_mat()

    def get_rk4_mat(self):
        rk4_mat = np.copy(self.u)

        def der(row):
            new_row = np.zeros(row.shape[0])
            for i in range(1, row.shape[0] - 1):
                new_row[i] = self.alpha / self.dx ** 2 * \
                             (row[i + 1] - 2 * row[i] + row[i - 1])
            return new_row

        for k in range(1, self.maxk):
            k1 = der(rk4_mat[:, k - 1])
            k2 = der(rk4_mat[:, k - 1] + k1 * self.dt / 2)
            k3 = der(rk4_mat[:, k - 1] + k2 * self.dt / 2)
            k4 = der(rk4_mat[:, k - 1] + k3 * self.dt)
            phi = (k1 + 2 * k2 + 2 * k3 + k4) / 6
            rk4_mat[:, k] = rk4_mat[:, k - 1] + phi * self.dt

        self.rk4_mat = rk4_mat
        return rk4_mat

    def rk4(self):
        if self.rk4_mat is not None:
            return self.rk4_mat

        return self.get_rk4_mat()

    def solution(self, x, t):
        lbd = np.pi / self.L
        return np.sin(x * lbd) * np.exp(-t * self.alpha * lbd ** 2)

    def get_sol_mat(self):
        sol_mat = np.zeros((self.n + 1, self.maxk))
        for i in range(self.n):
            for k in range(self.maxk):
                sol_mat[i, k] = self.solution(i / self.n * self.L, k / self.maxk * self.T)
        return sol_mat

    def sol_diff(self, mat):
        return np.abs(self.sol_mat - mat)

    def diff_norms(self):
        diff = [self.sol_diff(self.expl_euler()),
                self.sol_diff(self.impl_euler()),
                self.sol_diff(self.trapz_euler()),
                self.sol_diff(self.rk4())]
        names = ['expl', 'impl', 'trapz', 'rk4']
        norms = []
        for d in diff:
            norms.append(np.linalg.norm(d))
        res = dict(zip(names, norms))
        return res

    def diff_heights(self):
        diff = [self.sol_diff(self.expl_euler()),
                self.sol_diff(self.impl_euler()),
                self.sol_diff(self.trapz_euler()),
                self.sol_diff(self.rk4())]
        names = ['expl', 'impl', 'trapz', 'rk4']
        heights = []
        for d in diff:
            heights.append(np.max(d))
        res = dict(zip(names, heights))
        return res

    def plot_method(self, axis, X, Y, mat, title='', xlabel='Time', ylabel='L', zlabel='T', fontsize=12):
        axis.plot_surface(X, Y, mat, cmap='viridis', edgecolor='none')
        axis.set_title(title)
        axis.set_xlabel(xlabel, fontsize=fontsize)
        axis.set_ylabel(ylabel, fontsize=fontsize)
        axis.set_zlabel(zlabel, fontsize=fontsize)

    def plot_solution(self):
        X = np.linspace(0, self.T, self.maxk)
        Y = np.linspace(0, self.L, self.n + 1)
        X, Y = np.meshgrid(X, Y)

        figure, axis = plt.subplots(subplot_kw={'projection': '3d'}, figsize=(6, 6))
        self.plot_method(axis, X, Y, self.sol_mat, 'Аналитическое решение')

    def plot_fdm_solutions(self):
        X = np.linspace(0, self.T, self.maxk)
        Y = np.linspace(0, self.L, self.n + 1)
        X, Y = np.meshgrid(X, Y)

        figure, axis = plt.subplots(2, 2, subplot_kw={'projection': '3d'}, figsize=(8, 8))
        plt.subplots_adjust(hspace=0.2)

        self.plot_method(axis[0, 0], X, Y, self.expl_euler(), title='Explicit Euler')
        self.plot_method(axis[0, 1], X, Y, self.impl_euler(), title='Implicit Euler')
        self.plot_method(axis[1, 0], X, Y, self.trapz_euler(), title='Trapezoidal Euler')
        self.plot_method(axis[1, 1], X, Y, self.rk4(), title='4-th order Runge-Kutta')

        plt.suptitle('Решения уравнения численными методами', fontsize=16)
        plt.show()

    def plot_diff(self):
        X = np.linspace(0, self.T, self.maxk)
        Y = np.linspace(0, self.L, self.n + 1)
        X, Y = np.meshgrid(X, Y)

        figure, axis = plt.subplots(2, 2, subplot_kw={'projection': '3d'}, figsize=(8, 8))
        plt.subplots_adjust(hspace=0.2)

        self.plot_method(axis[0, 0], X, Y, self.sol_diff(self.expl_euler()), title='Explicit Euler')
        self.plot_method(axis[0, 1], X, Y, self.sol_diff(self.impl_euler()), title='Implicit Euler')
        self.plot_method(axis[1, 0], X, Y, self.sol_diff(self.trapz_euler()), title='Trapezoidal Euler')
        self.plot_method(axis[1, 1], X, Y, self.sol_diff(self.rk4()), title='4-th order Runge-Kutta')

        plt.suptitle('Отличия решений методом сеток от аналитического', fontsize=16)
        plt.show()

    def get_time_bench(self):
        def time_bench(f):
            start = timer()
            f()
            return timer() - start

        names = ['expl', 'impl', 'trapz', 'rk4']
        times = [time_bench(self.get_expl_mat),
                 time_bench(self.get_impl_mat),
                 time_bench(self.get_trapz_mat),
                 time_bench(self.get_rk4_mat)]
        return dict(zip(names, times))

    def get_memory_bench(self):
        def memory_bench(f):
            tracemalloc.start()
            f()
            start, end = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            return end - start

        names = ['expl', 'impl', 'trapz', 'rk4']
        times = [memory_bench(self.get_expl_mat),
                 memory_bench(self.get_impl_mat),
                 memory_bench(self.get_trapz_mat),
                 memory_bench(self.get_rk4_mat)]
        return dict(zip(names, times))


def fdm_group_plot(test_batch, method_name, title=''):
    df_options = []
    for fdm in test_batch:
        col_name = 'k = ' + str(fdm.maxk) + ',\n n = ' + str(fdm.n)
        vals = getattr(fdm, method_name)()
        for name in ['expl', 'impl', 'trapz', 'rk4']:
            df_options.append([name, col_name, vals[name]])

    df = pd.DataFrame(df_options, columns=['Алгоритм', 'Эксперимент', 'Значение'])
    df.pivot("Эксперимент", "Алгоритм", "Значение").plot(kind='bar')

    plt.title(title)
    plt.show()


def norm_group_plot(test_batch):
    fdm_group_plot(test_batch, 'diff_norms', title='Сравнение норм матриц различий')


def height_group_plot(test_batch):
    fdm_group_plot(test_batch, 'diff_norms', title='Наибольшее различие решения с аналитическим')


def plot_time_bench(test_batch):
    fdm_group_plot(test_batch, 'diff_norms', title='Сравнение времени работы алгоритмов')


def plot_memory_bench(test_batch):
    fdm_group_plot(test_batch, 'diff_norms', title='Сравнение аллоцируемой памяти компьютера')
