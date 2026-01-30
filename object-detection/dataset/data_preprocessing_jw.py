from imports import *
base_path = os.path.dirname(os.path.abspath(__file__))
anno_path = os.path.join(base_path, "train_annotation.json")

with open(anno_path, 'r') as f:
    annotations = json.load(f)


print("----------------------")
print(f"image_EA: {len(annotations)}")
print(f"")
