import os

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_dir = os.path.join(root_path, "models/checkpoints")
other_model_dir = os.path.join(root_path, "models/other")
