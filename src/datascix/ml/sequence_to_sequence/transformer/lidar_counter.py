import numpy as np
from utilix.data.storage.kind.file.pkl.pkl import Pkl
from utilix.os.file_system.file.file import File
from utilix.os.file_system.path.path import Path

# Path to your pickle file
path = Path(
    "/home/donkarlo/Dropbox/phd/data/experiements/oldest/robots/uav1/"
    "mind/memory/long_term/explicit/episodic/normal/"
    "lidar_scan_ranges_sliced_from_1_to_300000/"
    "lidar_scan_ranges_sliced_from_1_to_300000.pkl"
)

# Load pickle
os_file = File.init_from_path(path)
pk = Pkl(os_file, False)
sliced = pk.load()

# Extract objects
vals = sliced.get_values()

print("Number of LiDAR scans in the pickle:", len(vals))
