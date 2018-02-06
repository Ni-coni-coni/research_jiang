import multiprocessing as mp
import csv
with open("cpu_count.txt", "w") as file:
    csv_writer =  csv.writer(file, delimiter = ",")
    csv_writer.writerow(mp.cpu_count())
file.close()