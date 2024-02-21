import os
from tqdm import tqdm
import csv

SOURCE_ROOT = "/home/jay/savespace/database/generic"
TARGET_ROOT = "/home/jay/savespace/database/SYMLINKS/project_tailedcore/visa/"

def restructure_visa():
    global SOURCE_ROOT, TARGET_ROOT

    source_dir = os.path.join(SOURCE_ROOT, "visa")
    target_dir = os.path.join(TARGET_ROOT, "full")

    os.makedirs(target_dir, exist_ok = False)

    with open(os.path.join(source_dir, "split_csv", "1cls.csv")) as file:
        csvreader = csv.reader(file)
        header = next(csvreader)

        for row_no, row_content in tqdm(enumerate(csvreader)):
            
            class_, split_, label_, image_source, mask_source = row_content
            img_no = image_source.split('/')[-1]
            mask_no = mask_source.split('/')[-1] if mask_source != "" else ""
            label_ = "good" if label_=="normal" else "anomaly"

            if not os.path.exists(os.path.join(target_dir, class_, split_, label_)):
                os.makedirs(os.path.join(target_dir, class_, split_, label_), exist_ok=True)
            image_source = os.path.join(source_dir, image_source)
            image_target = os.path.join(target_dir, class_, split_, label_, img_no)
            os.symlink(image_source, image_target)

            if mask_source == "":
                continue

            else:
                assert split_=="test" and label_=="anomaly"

                if not os.path.exists(os.path.join(target_dir, class_, "ground_truth", label_)):
                    os.makedirs(os.path.join(target_dir, class_, "ground_truth", label_), exist_ok=True)
                mask_source = os.path.join(source_dir, mask_source)
                mask_target = os.path.join(target_dir, class_, "ground_truth", label_, mask_no)
                os.symlink(mask_source, mask_target)





if __name__ == "__main__":
    restructure_visa()