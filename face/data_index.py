import numpy as np
import csv
from collections import namedtuple
import json
import torch
import torch.nn.utils.rnn as rnn_utils
from tqdm import trange


tasks = ["_Balura_Game", "_Fixations", "_Reading",  "_Video_1",  "_Video_2"]
tasks_code = ["BLG", "FXS", "TEX", "VD1", "VD2"]

DataInfo = namedtuple("DataInfo", ["dimension_size", "people_size", "feature_size"])
exceptions = np.array([ 28,  36,  50,  56,  75,  82, 104, 105, 112, 124, 160, 205, 324])
amt_people = 335
info = DataInfo(dimension_size=2*len(tasks), people_size=amt_people-(exceptions<=amt_people).sum(), feature_size=3)

print("Dataset info:", info)
config = json.load(open("config.json"))
torch.manual_seed(config['seed'])

n = info.dimension_size * info.people_size
VAL = int(config['val'] * n)
TEST = int(config['test'] * n)
TRAIN = n - VAL - TEST

VAL_PEOPLE = int(config['val'] * info.people_size)
TEST_PEOPLE = int(config['test'] * info.people_size)
TRAIN_PEOPLE= info.people_size - VAL_PEOPLE - TEST_PEOPLE

def permutate(n):
	#n = info.people_size * info.dimension_size
	return torch.randperm(n)


TRAIN_PERM = permutate(TRAIN)
VAL_PERM = permutate(VAL)
TEST_PERM = permutate(TEST)


def data_index(person, dim):
    """
    Output sequence of eye gaze (x, y) positions from the dataset for a person and a dimension of that person (task, session, etc)
    Index starts at 0.
    The vectors are [x, y, flag], flag being if it's null
    """
    session = "S1" if dim % 2 == 0 else "S2"
    # S1_Balura_Game  S1_Fixations  S1_Horizontal_Saccades  S1_Random_Saccades  S1_Reading  S1_Video_1  S1_Video_2
    for exc in exceptions:
        person += (exc-1 <= person)
    num = str(person+1).rjust(3, "0")
    #global info, tasks, tasks_code
    dir = "data/Round_1/id_1" + num + "/" + session + "/" + session + tasks[dim//2] + \
        "/S_1" + num + "_" + session + "_" + tasks_code[dim//2] + \
        ".csv"

    pos = []
    mask = []

    with open(dir) as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        vecs = []
        pads = []
        for i, row in enumerate(spamreader):
            if i < 1:
                continue
            row = ''.join(row).split(",")
            if (i-1) % config['Hz'] == 0 and (i-1) != 0:
                vecs = np.stack(vecs)
                pads = np.stack(pads)
                pos.append(vecs)
                mask.append(pads)
                vecs = []
                pads = []
            if (i-1) % (config['Hz'] // config['second_split']) == 0:
                flag = (row[1] == 'NaN' or row[2] == 'NaN')
                vecs.append(np.array([float(row[1]), float(row[2]), flag]))
                arr = np.array([0]*(info.feature_size-1)+[info.feature_size]) if flag else np.ones(info.feature_size)
		# the info.feature_size instead of 1 is to rescale and give it equal "weight"
                pads.append(arr)
    pos=np.stack(pos)
    mask=np.stack(mask)
    return pos, mask, [tasks[dim//2]]  # pos + label/s

def normalize(dataset, mask):
    dat = dataset[VAL_PEOPLE+TEST_PEOPLE:, :, :, :-1] #* mask[VAL_PEOPLE+TEST_PEOPLE:, :, :, :-1]
    dat[torch.isnan(dat)] = 0.0
    N = mask.sum() / (info.feature_size-1)
    mean = dat.sum((0, 1, 2)) / N
    mean_ = mean[None, None, None]
    std = ((dat-mean)**2).sum((0, 1, 2)) / (N-1)
    std_ = std[None, None, None]
    print("mean: ", mean_.squeeze().cpu().numpy())
    print("std: ", std_.squeeze().cpu().numpy())
    dataset[:, :, :, :-1] -= mean
    dataset[:, :, :, :-1] /= (std)
    #dataset[:, :, :, :-1] *= 2 # scaling factor as std may push the values too low
    return dataset


def get_dataset(device = "cuda"):
    # dataset = torch.empty((info.people_dim, info.dimension_size, ))
    dataset=[]
    mask=[]
    lens=[]
    global info
    for idx in trange(info.people_size*info.dimension_size):
        # make the indexing work as the thing is flipped?
        pp, dim = idx % info.people_size, idx // info.people_size
        pos, mm, labels=data_index(pp, dim)
        pos=torch.as_tensor(pos, device = torch.device(device), dtype = torch.float32)
        mm=torch.as_tensor(mm, device = torch.device(device), dtype = torch.float32)
        dataset.append(pos)
        mask.append(mm)
        lens.append(pos.size()[0])

    dataset=rnn_utils.pad_sequence(dataset, batch_first=True)
    mask=rnn_utils.pad_sequence(mask, batch_first=True)
    print("Dataset tensor size:", dataset.size())
    if config['norm']:
        dataset=normalize(dataset, mask)
    return dataset, mask


def get_batch(i, dataset, mask, type = 0, last = False):
    # 0 -> train, 1 -> val, 2 -> test
    # get batch, as a packed padded sequence

    if type == 0:
        start=VAL + TEST
        end=n
        size=n - VAL - TEST
    elif type == 1:
        start=TEST
        end=TEST + VAL
        size=VAL
    else:
        start=0
        end=TEST
        size=TEST

    lens=[]
    if last:
        idxs=PERM[end-(size % batch_size):end]
    else:
        idxs=PERM[start*(i+1):start*(i+1)+config['batch_size']]
    batch=[]
    for idx in idxs:
        dat=dataset[idx//info.people_size, idx % info.people_size]
        batch.append(dat)
        lens.append(dat.size()[0])
    batch=rnn_utils.pad_sequence(batch, batch_first = True)
    # update mask with pad
    mask=rnn_utils.pad_sequence(mask, batch_first = True)

    # randomly permute the train batches at the end of an epoch
    if last:
        sum_=TEST+VAL
        PERM[sum_:]=PERM[sum_:][torch.randperm(n-sum_)]

    return batch, mask, lens

if __name__ == "__main__":
    print("Fetching dataset...")
    dataset, data_mask = get_dataset()
    print("Dataset fetched...")
    print("Saving to file...")
    torch.save(dataset, "data/dataset.pt")
    torch.save(dataset, "data/data_mask.pt")
    print("Dataset saved...")

