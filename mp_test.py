import multiprocessing as mp
import time

def count_down(n):
    print(n)
    while n > 0:
        n -= 1
    print(n)

def test_mp(p_num):
    processes = []

    for _ in range(p_num):
        p = mp.Process(target=count_down, args=(60000000,))
        p.start()
        processes.append(p)
        time.sleep(0.1)

    for p in processes:
        time.sleep(0.1)
        p.join()

if __name__ == '__main__':

    test_mp(5)
