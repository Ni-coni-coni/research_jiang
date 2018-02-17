import multiprocessing as mp
import time

def f(conn, n):
    conn.send([n, None, 'hello'])
    conn.close()

if __name__ == '__main__':

    processes = []

    for i in range(5):
        pipe_in, pipe_out = mp.Pipe()
        p = mp.Process(target=f, args=(pipe_out, i,))
        p.start()
        processes.append(p)
        print(pipe_in.recv())

    for p in processes:
        p.join()
