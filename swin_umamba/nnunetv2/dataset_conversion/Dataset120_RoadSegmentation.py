import multiprocessing
import shutil
from multiprocessing import Pool

from batchgenerators.utilities.file_and_folder_operations import *

from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw
from skimage import io
from acvl_utils.morphology.morphology_helper import generic_filter_components
from scipy.ndimage import binary_fill_holes


def load_and_covnert_case(input_image: str, input_seg: str, output_image: str, output_seg: str,
                          min_component_size: int = 50):
    seg = io.imread(input_seg)
    seg[seg == 255] = 1
    image = io.imread(input_image)
    #print(image.shape)
    image = image.sum(axis=2)
    mask = image == (3 * 255)
    # print(mask.shape)
    # the dataset has large white areas in which road segmentations can exist but no image information is available.
    # Remove the road label in these areas
    mask = generic_filter_components(mask, filter_fn=lambda ids, sizes: [i for j, i in enumerate(ids) if
                                                                         sizes[j] > min_component_size])
    mask = binary_fill_holes(mask)
    seg[mask] = 0
    #print(seg.shape)
    io.imsave(output_seg, seg, check_contrast=False)
    shutil.copy(input_image, output_image)


if __name__ == "__main__":
    # extracted archive from https://www.kaggle.com/datasets/insaff/massachusetts-roads-dataset?resource=download
    source = "/home/cwq/MedicalDP/SwinUmamba/data/nnUNet_raw/Task704_Endovis17"

    dataset_name = 'Dataset704_Endovis17'

    imagestr = join(nnUNet_raw, dataset_name, 'imagesTr')
    imagests = join(nnUNet_raw, dataset_name, 'imagesTs')
    labelstr = join(nnUNet_raw, dataset_name, 'labelsTr')
    labelsts = join(nnUNet_raw, dataset_name, 'labelsTs')
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(imagests)
    maybe_mkdir_p(labelstr)
    maybe_mkdir_p(labelsts)

    # train_source = join(source,'train')
    # test_source = join(source,'test')
    # join(source, 'training'))
    #join(source, 'testing')

    with multiprocessing.get_context("spawn").Pool(8) as p:

        # not all training images have a segmentation
        valid_ids = subfiles(join(source, 'labelsTr'), join=False, suffix='png')
        # print("valid_ids",valid_ids)
        num_train = len(valid_ids)
        print("train numbers:",num_train)
        r = []
        for v in valid_ids:
            r.append(
                p.starmap_async(
                    load_and_covnert_case,
                    ((
                         join(source, 'imagesTr', v),
                         join(source, 'labelsTr', v),
                         join(imagestr, v[:-4] + '_0000.png'),
                         join(labelstr, v),
                         50
                     ),)
                )
            )
            print("Train dataset finished")

        # test set
        valid_ids = subfiles(join(source, 'labelsTs'), join=False, suffix='png')
        nums_test = len(valid_ids)
        print("test numbers:", nums_test)
        for v in valid_ids:
            r.append(
                p.starmap_async(
                    load_and_covnert_case,
                    ((
                         join(source, 'imagesTs', v),
                         join(source, 'labelsTs', v),
                         join(imagests, v),
                         join(labelsts, v),
                         50
                     ),)
                )
            )
            #print("Test dataset finished")
        _ = [i.get() for i in r]

    generate_dataset_json(join(nnUNet_raw, dataset_name), {0: 'R', 1: 'G', 2: 'B'}, {"background": 0, "Nodule ": 1},
                          num_train, '.png', dataset_name=dataset_name)
    print('Done')
