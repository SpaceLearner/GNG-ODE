import pickle as pkl

with open("train_timestamp.txt", "r") as fp:
    lines = fp.readlines()
    total_diff     = 0
    total_interval = 0
    for idx, line in enumerate(lines):
        words = line.strip().split(",")
        times = list(map(float, words))
       #  print(times)
        for i in range(1, len(times)):
            total_diff += (times[i] - times[i-1])
            total_interval += 1
        
    print(total_diff / total_interval)