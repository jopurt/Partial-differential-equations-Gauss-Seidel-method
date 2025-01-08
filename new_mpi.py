from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Вычисляет локальные размеры сетки и стартовые строки для каждого процесса
def compute_local_grid_sizes(N, size):
    rows_per_proc = N // size
    extra_rows = N % size

    # start_row_list = np.zeros(size, dtype=int)
    # local_nrows_list = np.zeros(size, dtype=int)
    #
    # for r in range(size):
    #     if r < extra_rows:
    #         local_nrows = rows_per_proc + 1
    #         start_row = r * local_nrows
    #     else:
    #         local_nrows = rows_per_proc
    #         start_row = r * local_nrows + extra_rows
    #     start_row_list[r] = start_row
    #     local_nrows_list[r] = local_nrows

    # векторизация
    processes = np.arange(size)
    local_nrows_list = rows_per_proc + (processes < extra_rows)
    start_row_list = np.cumsum(np.insert(local_nrows_list[:-1], 0, 0))

    return start_row_list, local_nrows_list

# Инициализирует локальные массивы и задает граничные условия
def initialize_arrays(local_nrows, N):
    u_local = np.zeros((local_nrows + 2, N + 2))
    u_old = np.zeros_like(u_local)
    f_local = np.ones_like(u_local)  # Инициализация правой части

    # Граничные условия
    u_local[:, 0] = 0.0  # Левая граница
    u_local[:, -1] = 0.0  # Правая граница

    return u_local, u_old, f_local

# Обменивается граничными строками с соседними процессами
def exchange(u_local, comm, up, down):
    if up != MPI.PROC_NULL:
        comm.Sendrecv(u_local[1, :], dest=up, sendtag=0,
                      recvbuf=u_local[0, :], source=up, recvtag=1)
    if down != MPI.PROC_NULL:
        comm.Sendrecv(u_local[-2, :], dest=down, sendtag=1,
                      recvbuf=u_local[-1, :], source=down, recvtag=0)


def gauss_seidel(u_local, u_old, f_local, local_nrows, N, h2, up, down, max_iter, eps, comm, rank):
    for iteration in range(max_iter):
        exchange(u_local, comm, up, down)  # Обмен границ между процессами
        np.copyto(u_old, u_local)  # Сохраняем предыдущее состояние

        # Чередующийся порядок обновления
        if iteration % 2 == 0:
            # Прямой порядок
            for i in range(1, local_nrows + 1):
                for j in range(1, N + 1):
                    u_local[i, j] = 0.25 * (u_local[i - 1, j] + u_local[i + 1, j] +
                                            u_local[i, j - 1] + u_local[i, j + 1] -
                                            h2 * f_local[i, j])
        else:
            # Обратный порядок
            for i in range(local_nrows, 0, -1):
                for j in range(N, 0, -1):
                    u_local[i, j] = 0.25 * (u_local[i - 1, j] + u_local[i + 1, j] +
                                            u_local[i, j - 1] + u_local[i, j + 1] -
                                            h2 * f_local[i, j])

        # Вычисление нормы разности
        diff_local = np.sum((u_local[1:-1, 1:-1] - u_old[1:-1, 1:-1]) ** 2)
        diff_global = comm.allreduce(diff_local, op=MPI.SUM)  # Суммируем разности по всем процессам
        diff_norm = np.sqrt(diff_global)

        # if rank == 0 and iteration % 10 == 0:
        #     print(f"Iteration {iteration}, diff_norm = {diff_norm:.6e}")

        if diff_norm < eps:
            # if rank == 0:
            #     print(f"Converged after {iteration} iterations with diff_norm = {diff_norm:.6e}")
            break


def red_black_gauss_seidel(u_local, f_local, local_nrows, N, h2, up, down, max_iter, eps, comm, rank):
    for iteration in range(max_iter):
        exchange(u_local, comm, up, down)

        # Создание масок для красных и черных точек
        row_indices = np.arange(1, local_nrows + 1)[:, np.newaxis]
        col_indices = np.arange(1, N + 1)
        red_mask = (row_indices + col_indices) % 2 == 0
        black_mask = ~red_mask

        # Обновление "красных" точек
        u_local[1:-1, 1:-1][red_mask] = 0.25 * (
            u_local[:-2, 1:-1][red_mask] +
            u_local[2:, 1:-1][red_mask] +
            u_local[1:-1, :-2][red_mask] +
            u_local[1:-1, 2:][red_mask] -
            h2 * f_local[1:-1, 1:-1][red_mask]
        )

        # Обмен границами снова перед обновлением черных точек
        exchange(u_local, comm, up, down)

        # Обновление "черных" точек
        u_local[1:-1, 1:-1][black_mask] = 0.25 * (
            u_local[:-2, 1:-1][black_mask] +
            u_local[2:, 1:-1][black_mask] +
            u_local[1:-1, :-2][black_mask] +
            u_local[1:-1, 2:][black_mask] -
            h2 * f_local[1:-1, 1:-1][black_mask]
        )

        # Вычисление локальной нормы разности
        diff_local = np.sum((u_local[1:-1, 1:-1] - f_local[1:-1, 1:-1]) ** 2)
        diff_global = comm.allreduce(diff_local, op=MPI.SUM)
        diff_norm = np.sqrt(diff_global)

        # Проверка сходимости
        if diff_norm < eps:
            if rank == 0:
                print(f"Converged in {iteration} iterations")
            break




def solution(u_local, local_nrows_list, start_row_list, N, comm, rank, size):
    local_nrows = local_nrows_list[rank]
    u_local_flat = u_local[1:-1, 1:-1].flatten()

    counts = local_nrows_list * N
    displs = np.insert(np.cumsum(counts), 0, 0)[:-1]

    u_global = None
    if rank == 0:
        u_global = np.empty(np.sum(counts), dtype=np.float64)

    comm.Gatherv(u_local_flat, [u_global, counts, displs, MPI.DOUBLE], root=0)

    if rank == 0:
        u = np.zeros((N, N))
        current = 0
        for r in range(size):
            nrows = local_nrows_list[r]
            srow = start_row_list[r]
            u[srow:srow + nrows, :] = u_global[current:current + nrows * N].reshape((nrows, N))
            current += nrows * N

        return u
    else:
        return None


# Инициализация MPI
comm = MPI.COMM_WORLD
global_rank = comm.Get_rank()
global_size = comm.Get_size()

N = 100
max_iter = 1000
eps = 1e-6



processes_list = []
times_list = []

for num_procs in range(1, global_size + 1):
    color = 1 if global_rank < num_procs else MPI.UNDEFINED
    new_comm = comm.Split(color, global_rank)
    if color == MPI.UNDEFINED:
        continue

    rank = new_comm.Get_rank()
    size = new_comm.Get_size()

    h = 1.0 / (N + 1)
    h2 = h * h

    # Вычисление локальных размеров сетки
    start_row_list, local_nrows_list = compute_local_grid_sizes(N, size)
    start_row = start_row_list[rank]
    local_nrows = local_nrows_list[rank]

    # Определение соседей
    up = rank - 1 if rank > 0 else MPI.PROC_NULL
    down = rank + 1 if rank < size - 1 else MPI.PROC_NULL

    # Инициализация массивов
    u_local, u_old, f_local = initialize_arrays(local_nrows, N)

    # Синхронизация процессов перед началом измерения времени
    new_comm.Barrier()
    start_time = MPI.Wtime()

    # Основной итерационный цикл
    # gauss_seidel(u_local, u_old, f_local, local_nrows, N, h2, up, down, max_iter, eps, new_comm, rank)
    red_black_gauss_seidel(u_local, f_local, local_nrows, N, h2, up, down, max_iter, eps, new_comm, rank)

    # Синхронизация процессов после окончания вычислений
    new_comm.Barrier()
    end_time = MPI.Wtime()
    elapsed_time = end_time - start_time

    # Решение
    u = solution(u_local, local_nrows_list, start_row_list, N, new_comm, rank, size)

    if rank == 0:
        processes_list.append(num_procs)
        times_list.append(elapsed_time)

    new_comm.Free()

# Построение графиков на корневом процессе
if global_rank == 0:
    # Расчет ускорения и эффективности
    T1 = times_list[0]

    speedups = [T1 / t for t in times_list]
    # print(speedups,processes_list)
    efficiencies = [s / p for s, p in zip(speedups, processes_list)]

    print('sdfsdfs',times_list)
    print('AaaA',sum(times_list))

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    axs[0, 0].plot(processes_list, times_list, marker='o', linestyle='-', color='b')
    axs[0, 0].set_title('Время выполнения от числа процессов')
    axs[0, 0].set_xlabel('Количество процессов')
    axs[0, 0].set_ylabel('Время выполнения (секунды)')
    axs[0, 0].grid(True)

    sns.heatmap(u, ax=axs[0, 1], cmap='viridis', cbar=True)
    axs[0, 1].set_title('Тепловая карта результирующей матрицы')
    axs[0, 1].set_xlabel('X')
    axs[0, 1].set_ylabel('Y')

    axs[1, 0].plot(processes_list, speedups, marker='o', linestyle='-', color='g')
    axs[1, 0].set_title('Ускорение')
    axs[1, 0].set_xlabel('Количество процессов')
    axs[1, 0].set_ylabel('Ускорение')
    axs[1, 0].grid(True)

    axs[1, 1].plot(processes_list, efficiencies, marker='o', linestyle='-', color='r')
    axs[1, 1].set_title('Эффективность')
    axs[1, 1].set_xlabel('Количество процессов')
    axs[1, 1].set_ylabel('Эффективность')
    axs[1, 1].grid(True)

    plt.suptitle('Анализ производительности', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    print(u)
    # print(eps)

MPI.Finalize()