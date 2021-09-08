import os
import argparse
import cv2
import numpy as np
import nibabel as nib
from glob import glob
from multiprocessing.dummy import Pool


parser = argparse.ArgumentParser(description='Slice Maker')
parser.add_argument('--in_path', type=str, default='/data/kits19/data')
parser.add_argument('--out_path', type=str, default='/data/kits/train')
parser.add_argument('--process_num', type=int, default=10)
parser.add_argument('--dataset', type=str, default='kits', choices=['kits', 'lits'])
parser.add_argument('--task', type=str, default='tumor', choices=['tumor', 'organ'])
parser.add_argument('--mode', type=str, default='train')

args = parser.parse_args()


def main():
    if not os.path.exists(args.out_path):
        os.mkdir(args.out_path)
    list_path = os.path.join(args.out_path, args.mode)
    if not os.path.exists(list_path):
        os.mkdir(list_path)

    if args.dataset == 'kits':
        paths = glob(os.path.join(args.in_path, "case_*/imaging*.nii.gz"))
    elif args.dataset == 'lits':
        paths = glob(os.path.join(args.in_path, "volume-*.nii"))

    pool = Pool(args.process_num)
    result = pool.map(make_slice, paths)

    # Save info
    tumor_slices = []
    for i in result:
        tumor_slices += i
    np.save(os.path.join(list_path, '%s_%s_slices.npy' % (args.dataset, args.task)), tumor_slices)


def make_slice(path):
    """
    Cut 3D kits data into 2D slices
    :param path: /*/*.nii.gz
    :return: Slices and Infos
    """
    if args.dataset == 'kits':
        case, vol, seg = read_kits(path)
    elif args.dataset == 'lits':
        case, vol, seg = read_lits(path)
    result = []
    for i in range(vol.shape[0]):
        ct_slice = vol[i, ...]
        if ct_slice.shape != [512, 512]:
            ct_slice = cv2.resize(ct_slice, dsize=(512, 512), interpolation=cv2.INTER_LINEAR)
        mask_slice = seg[i, ...]
        np.savez_compressed(f'{args.out_path}/{args.mode}/{case}_{i}.npz', ct=ct_slice, mask=mask_slice)
        if args.task == 'organ':
            if np.any(mask_slice > 0):
                result.append(f'{case}_{i}.npz')
        elif args.task == 'tumor':
            if np.any(mask_slice > 1):
                result.append(f'{case}_{i}.npz')

    print(f'complete making slices of {case}')
    return result


def read_kits(path):
    dir = os.path.dirname(path)
    vol = nib.load(path).get_fdata()
    seg = nib.load(os.path.join(dir, 'segmentation.nii.gz')).get_fdata().astype('int8')
    case = os.path.split(dir)[-1][-5:]
    return case, vol, seg


def read_lits(path):
    vol = nib.load(path).get_fdata()
    seg = nib.load(path.replace('volume', 'segmentation')).get_fdata().astype('int8')
    case = path.split('-')[-1].split('.')[0]
    vol = np.transpose(vol, (2, 0, 1))
    seg = np.transpose(seg, (2, 0, 1))
    return case, vol, seg


if __name__ == '__main__':
    main()
