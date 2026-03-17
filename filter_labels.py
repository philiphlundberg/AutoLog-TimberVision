import os

dataset_path = "/home/philiph/Documents/DigitalFysik/TimberVision/timbervision/yolo_training/dataset/"
labels_path = os.path.join(dataset_path, "labels")

for root, dirs, files in os.walk(labels_path):
    for fname in files:
        if not fname.endswith(".txt"):
            continue
        fpath = os.path.join(root, fname)
        with open(fpath, "r") as f:
            lines = f.readlines()
        kept = []
        for line in lines:
            cls = int(line.split()[0])
            if cls == 0:  # Cut → stays as 0
                kept.append(line)
            elif cls == 2:  # Side → remap to 1
                kept.append("1" + line[1:])
            # cls 1 (Bound) and 3 (Trunk) are dropped
        with open(fpath, "w") as f:
            f.writelines(kept)