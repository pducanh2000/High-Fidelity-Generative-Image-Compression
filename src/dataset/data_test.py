import os
from glob import glob

kodak_data_folder = "./../../data/kodak_dataset/"       # Use this dataset for evaluating
vimeo_data_folder = "./../../data/vimeo_interp_test/"   # Use this dataset for training


def print_evaluating_images(data_folder):
    kodak_image_list = glob(os.path.abspath(os.path.join(kodak_data_folder, "*.png")))
    count = 0
    for image_name in kodak_image_list:
        print(image_name)
        count += 1
    print("\n\nTotal {} images in Kodak dataset".format(count))


def test_vimeo_data(vimeo_folder):
    vimeo_images_list = glob(os.path.abspath(os.path.join(vimeo_folder, "*/*/*/*.png")))

    for i, image_name in enumerate(vimeo_images_list):
        if (i+1) % 10 == 0:
            print(f"Eg image path: {image_name}")
    print("Total {} images in vimeo data folder".format(len(vimeo_images_list)))


if __name__ == "__main__":
    test_vimeo_data(vimeo_data_folder)




