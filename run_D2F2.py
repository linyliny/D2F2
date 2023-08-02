import h5py
import torch
from pathlib import Path
import matplotlib.pyplot as plt

from fastmri.data.subsample import create_mask_for_mask_type
from fastmri.data.transforms import VarNetDataTransform
from fastmri.data import transforms
from utils import retrieve_metadata
from D2F2_model import D2F2


# This code is based on the official code of fastMRI. Please make sure to correctly import the relevant fastmri code before running it.


# choose which h5 file and slice you want to reconstruct
data_path = '***/file1000926.h5'
result_restore_path = '***/output.png'
fname = Path(data_path)
dataslice = 20

# give model checkpoint path
check_point_path = '***/D2F2_6_4.ckpt'

# define some parameters and model.
mask = create_mask_for_mask_type("random", [0.08], [4])
transform = VarNetDataTransform(mask_func=mask)
net = D2F2()

# load the model
state_dicts = torch.load(check_point_path)['state_dict']
state_dicts_new = {}
for key in state_dicts:
    if key == 'loss.w':
        continue
    value = state_dicts[key]
    lenghOfFirstElement = len(key.split('.')[0]) + 1
    state_dicts_new[key[lenghOfFirstElement:]] = value
net.load_state_dict(state_dicts_new)
net.cuda()
net.eval()

# read file's info
metadata, num_slices = retrieve_metadata(data_path)

# read data and reconstruct it
with torch.no_grad():
    with h5py.File(fname, "r") as hf:
        kspace = hf["kspace"][dataslice]
        target = hf["reconstruction_rss"][dataslice]
        attrs = dict(hf.attrs)
        attrs.update(metadata)
        hf.close()

    sample = transform(kspace, None, target, attrs, fname.name, dataslice)
    masked_kspace = sample.masked_kspace.unsqueeze(0).cuda()
    mask = sample.mask.unsqueeze(0).cuda()
    num_low_freq = sample.num_low_frequencies

    output = net(masked_kspace, mask, num_low_freq).squeeze(0)
    target, output = transforms.center_crop_to_smallest(sample.target, output)

# visual the results
target_array, output_array = target.cpu().numpy(), output.cpu().numpy()
plt.figure(dpi=300)
plt.subplot(1,3,1)
plt.imshow(output_array, 'gray')
plt.axis('off')
plt.xticks([])
plt.yticks([])
plt.subplot(1,3,2)
plt.imshow(target_array, 'gray')
plt.axis('off')
plt.xticks([])
plt.yticks([])
plt.subplot(1,3,3)
plt.imshow(abs(target_array - output_array), 'gray')
plt.axis('off')
plt.xticks([])
plt.yticks([])

plt.savefig(result_restore_path, bbox_inches='tight')
