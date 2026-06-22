import os
import sys
import random

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utility import *  # noqa

def _groups_helper(
        image_folder, 
        groups, 
        label_folder=None, 
        landmark_folder=None,
        prefix=None, 
        suffix='.npy', 
        removezeros=True, 
        image_key='image', 
        label_key='label',
        landmark_key='landmark'
        ):
    if label_folder == None:
        without_label = True
    else:
        without_label = False
    if landmark_folder == None:
        without_landmark = True
    else:
        without_landmark = False
    
    image_list = subfiles(image_folder, join=False, prefix=prefix, suffix=suffix)
    image_list.sort(key=lambda x:int(re.findall('\\d+', x)[-1]))
    number_of_data = len(image_list)
    print(f"number of data: {number_of_data}")

    random.seed(1234)
    random.shuffle(image_list)

    output_dict = {}
    pointer1 = 0
    for key in groups:
        output_dict[key] = []
        pointer2 = pointer1 + groups[key]
        file_list = image_list[pointer1:pointer2]
        pointer1 = pointer2
        groups[key] = file_list

    for key in output_dict:    
        for file_name in groups[key]:
            image_name = join(image_folder, file_name)
            assert isfile(image_name), 'An image data does not exist. data name: '+image_name
            dict = {image_key: image_name,}
            # label
            if not without_label:
                if removezeros:
                    if file_name[-9:-4] == "_0000":
                        file_name = file_name[:-9] + file_name[-4:]
                    if file_name[-12:-7] == "_0000":
                        file_name = file_name[:-12] + file_name[-7:]
                label_name = join(label_folder, file_name)
                # label_name = join(label_folder, 'label'+file_name[3:])
                
                assert isfile(label_name), 'An label data does not exist. data name: '+label_name
                dict[label_key] = label_name
            # landmark
            if not without_landmark:
                if removezeros:
                    if file_name[-9:-4] == "_0000":
                        file_name = file_name[:-9] + file_name[-4:]
                    if file_name[-12:-7] == "_0000":
                        file_name = file_name[:-12] + file_name[-7:]
                if file_name.endswith('.nii') or file_name.endswith('.npy'):
                    landmark_name = join(landmark_folder, file_name[:-4]+'.txt')
                elif file_name.endswith('.nii.gz') or file_name.endswith('.npy'):
                    landmark_name = join(landmark_folder, file_name[:-7]+'.txt')
                
                assert isfile(landmark_name), 'An label data does not exist. data name: '+label_name
                dict[landmark_key] = landmark_name

            output_dict[key].append(dict)
    return output_dict

def group_LIDC():

    # LIDC-IDRI_15mm
    dataset_path = '/mnt/nvme1n1/mingrui/dataset/LIDC-IDRI_15mm/npy'
    split_file_path = '/home/mingrui/disk1/projects/20251103_DiffusionCorr/diffusioncorr/configs/LIDC_IDRI.json'
    image_folder = dataset_path
    len(subfiles(image_folder, suffix='.npy'))
    print(str(len(subfiles(image_folder, suffix='.npy'))))
    groups = {
        'train': 800,
        'test': 49
    }
    outdict1 = _groups_helper(image_folder, groups, label_folder=None, suffix='.npy', removezeros=False)
    output_dict = {**outdict1, }

    if isfile(split_file_path):
        os.remove(split_file_path)
    save_json(output_dict, split_file_path)

def group_AE_1mm():
    split_file_path = '/home/mingrui/disk1/projects/20251128_DiffusionCorrtest/diffusioncorr/configs/AE_1mm.json'

    # LIDC-IDRI_1mm
    dataset_path = '/mnt/nvme1n1/mingrui/dataset/LIDC-IDRI_1mm/npy'
    image_folder = dataset_path
    len(subfiles(image_folder, suffix='.npy'))
    print(str(len(subfiles(image_folder, suffix='.npy'))))
    groups = {
        'lidc': 849
    }
    outdict1 = _groups_helper(image_folder, groups, label_folder=None, suffix='.npy', removezeros=False)

    # flare2023 2200
    dataset_path = '/mnt/nvme2n1/mingrui/dataset/FLARE23_1mm/npy'
    image_folder = dataset_path
    len(subfiles(image_folder, suffix='.npy'))
    print(str(len(subfiles(image_folder, suffix='.npy'))))
    groups = {
        'flare2200': 2200
    }
    outdict2 = _groups_helper(image_folder, groups, label_folder=None, suffix='.npy', removezeros=False)

    # flare2023 1800
    dataset_path = '/mnt/nvme3n1/mingrui/dataset/FLARE23_1mm/npy'
    image_folder = dataset_path
    len(subfiles(image_folder, suffix='.npy'))
    print(str(len(subfiles(image_folder, suffix='.npy'))))
    groups = {
        'flare1800': 1800
    }
    outdict3 = _groups_helper(image_folder, groups, label_folder=None, suffix='.npy', removezeros=False)

    output_dict = {**outdict1, **outdict2, **outdict3}

    if isfile(split_file_path):
        os.remove(split_file_path)
    save_json(output_dict, split_file_path)

def group_AE_15mm():
    split_file_path = '/home/mingrui/disk1/projects/20260112_DiffusionCorr/miccai26/configs/AE_15mm.json'

    # LIDC-IDRI_1mm
    dataset_path = '/home/mingrui/ssd1/processed_dataset/LIDC-IDRI_15mm/npy'
    image_folder = dataset_path
    len(subfiles(image_folder, suffix='.npy'))
    print(str(len(subfiles(image_folder, suffix='.npy'))))
    groups = {
        'lidc': 833
    }
    outdict1 = _groups_helper(image_folder, groups, label_folder=None, suffix='.npy', removezeros=False)

    # flare2023 2200
    dataset_path = '/home/mingrui/ssd1/processed_dataset/Flare2023_15mm/npy'
    image_folder = dataset_path
    len(subfiles(image_folder, suffix='.npy'))
    print(str(len(subfiles(image_folder, suffix='.npy'))))
    groups = {
        'flare': 3884
    }
    outdict2 = _groups_helper(image_folder, groups, label_folder=None, suffix='.npy', removezeros=False)

    output_dict = {**outdict1, **outdict2}

    if isfile(split_file_path):
        os.remove(split_file_path)
    save_json(output_dict, split_file_path)

def group_4DCT():

    # DIRLab_4DCT_15mm
    dataset_path = '/home/mingrui/disk1/processed_dataset/DIRLab_4DCT_1mm'
    split_file_path = '/home/mingrui/disk1/projects/20260112_DiffusionCorr/miccai26/configs/4DCT_1mm.json'

    image_folder = join(dataset_path, 'npy')
    landmark_folder = join(dataset_path, 'landmarks')

    image_list = subfiles(image_folder, join=False, suffix='T00_s.npy')
    image_list.sort(key=lambda x:int(re.findall('\\d+', x)[-1]))
    number_of_data = len(image_list)
    print(f"number of data: {number_of_data}")
    
    output_dict = {'test':[]}
    for image_1_name in image_list:
        image_fullname_1 = join(image_folder, image_1_name)
        image_fullname_2 = join(image_folder, image_1_name[:-10] + '_T50_s.npy')
        landmark_fullname_1 = join(landmark_folder, image_1_name[:-10] + '_T00_s.txt')
        landmark_fullname_2 = join(landmark_folder, image_1_name[:-10] + '_T50_s.txt')
        output_dict['test'].append({'image_1':image_fullname_1, 'image_2':image_fullname_2, 'landmark_1':landmark_fullname_1, 'landmark_2':landmark_fullname_2})

    if isfile(split_file_path):
        os.remove(split_file_path)
    save_json(output_dict, split_file_path)

def group_LiverDIR():

    # DIRLab_4DCT_15mm
    dataset_path = '/home/mingrui/disk1/dataset/LiverDIR_1mm'
    split_file_path = '/home/mingrui/disk1/projects/20251103_DiffusionCorr/diffusioncorr/configs/LiverDIR_1mm.json'

    image_folder = join(dataset_path, 'npy')
    landmark_folder = join(dataset_path, 'landmarks')

    image_list = subfiles(image_folder, join=False, suffix='img1.npy')
    image_list.sort(key=lambda x:int(re.findall('\\d+', x)[-1]))
    number_of_data = len(image_list)
    print(f"number of data: {number_of_data}")
    
    output_dict = {'test':[]}
    for image_1_name in image_list:
        image_fullname_1 = join(image_folder, image_1_name)
        image_fullname_2 = join(image_folder, image_1_name[:-9] + '_img2.npy')
        landmark_fullname_1 = join(landmark_folder, image_1_name[:-9] + '_landmarks1.txt')
        landmark_fullname_2 = join(landmark_folder, image_1_name[:-9] + '_landmarks2.txt')
        output_dict['test'].append({'image_1':image_fullname_1, 'image_2':image_fullname_2, 'landmark_1':landmark_fullname_1, 'landmark_2':landmark_fullname_2})

    if isfile(split_file_path):
        os.remove(split_file_path)
    save_json(output_dict, split_file_path)

def group_DeepLesion():

    # DIRLab_4DCT_15mm
    dataset_path = '/home/mingrui/ssd1/processed_dataset/DeepLesion_1mm'
    split_file_path = '/home/mingrui/disk1/projects/20260112_DiffusionCorr/miccai26/configs/DeepLesion_1mm.json'

    image_folder = join(dataset_path, 'npy')
    landmark_folder = join(dataset_path, 'landmarks')

    output_dict = {'test':[], 'train':[], 'valid':[]}

    # test
    image_list = subfiles(image_folder, join=False, suffix='_source.npy', prefix='test')
    image_list.sort(key=lambda x:int(re.findall('\\d+', x)[-1]))
    number_of_data = len(image_list)
    print(f"number of data: {number_of_data}")
    for image_1_name in image_list:
        image_fullname_1 = join(image_folder, image_1_name)
        image_fullname_2 = join(image_folder, image_1_name[:-11] + '_target.npy')
        landmark_fullname_1 = join(landmark_folder, image_1_name[:-11] + '_source.txt')
        landmark_fullname_2 = join(landmark_folder, image_1_name[:-11] + '_target.txt')
        output_dict['test'].append({'image_1':image_fullname_1, 'image_2':image_fullname_2, 'landmark_1':landmark_fullname_1, 'landmark_2':landmark_fullname_2})

    # train
    image_list = subfiles(image_folder, join=False, suffix='_source.npy', prefix='train')
    image_list.sort(key=lambda x:int(re.findall('\\d+', x)[-1]))
    number_of_data = len(image_list)
    print(f"number of data: {number_of_data}")
    for image_1_name in image_list:
        image_fullname_1 = join(image_folder, image_1_name)
        image_fullname_2 = join(image_folder, image_1_name[:-11] + '_target.npy')
        landmark_fullname_1 = join(landmark_folder, image_1_name[:-11] + '_source.txt')
        landmark_fullname_2 = join(landmark_folder, image_1_name[:-11] + '_target.txt')
        output_dict['train'].append({'image_1':image_fullname_1, 'image_2':image_fullname_2, 'landmark_1':landmark_fullname_1, 'landmark_2':landmark_fullname_2})

    # valid
    image_list = subfiles(image_folder, join=False, suffix='_source.npy', prefix='valid')
    image_list.sort(key=lambda x:int(re.findall('\\d+', x)[-1]))
    number_of_data = len(image_list)
    print(f"number of data: {number_of_data}")
    for image_1_name in image_list:
        image_fullname_1 = join(image_folder, image_1_name)
        image_fullname_2 = join(image_folder, image_1_name[:-11] + '_target.npy')
        landmark_fullname_1 = join(landmark_folder, image_1_name[:-11] + '_source.txt')
        landmark_fullname_2 = join(landmark_folder, image_1_name[:-11] + '_target.txt')
        output_dict['valid'].append({'image_1':image_fullname_1, 'image_2':image_fullname_2, 'landmark_1':landmark_fullname_1, 'landmark_2':landmark_fullname_2})

    if isfile(split_file_path):
        os.remove(split_file_path)
    save_json(output_dict, split_file_path)

if __name__ == "__main__":
    # main_random_groups()
    group_4DCT()
    # group_AE_1mm()
    # group_AE_15mm()
    # group_LiverDIR()
    # group_DeepLesion()
