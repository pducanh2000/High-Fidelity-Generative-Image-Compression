import os
from glob import glob
from sklearn.model_selection import train_test_split
import json

# Use this for evaluating
kodak_data_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/kodak_dataset"))
# Use this for training
vimeo_data_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/vimeo_interp_test"))


def print_evaluating_images(data_folder=kodak_data_folder):
    kodak_image_list = glob(os.path.abspath(os.path.join(data_folder, "*.png")))
    count = 0
    for image_name in kodak_image_list:
        print(image_name)
        count += 1
    print("\n\nTotal {} images in Kodak dataset".format(count))


def test_vimeo_data(vimeo_folder=vimeo_data_folder):
    vimeo_images_list = glob(os.path.abspath(os.path.join(vimeo_folder, "*/*/*/*.png")))

    for i, image_name in enumerate(vimeo_images_list):
        if (i + 1) % 10 == 0:
            print(f"Eg image path: {image_name}")
    print("Total {} images in vimeo data folder".format(len(vimeo_images_list)))


def create_kodak_json(kodak_folder=kodak_data_folder):
    kodak_image_list = glob(os.path.join(kodak_folder, "*.png"))
    kodak_image_list += glob(os.path.join(kodak_folder, "*.jpg"))

    json_path = os.path.join(kodak_folder, "data.json")

    with open(json_path, "w") as f:
        json.dump(obj=kodak_image_list, fp=f)
    print(
        "Created Kodak json file!!!"
        f"\n{len(kodak_image_list)} images for evaluating"
        )
    print("-" * 50)
    return


def create_vimeo_json(
        vimeo_folder=os.path.abspath(vimeo_data_folder),
):
    vimeo_images_list = list(glob(os.path.abspath(os.path.join(vimeo_folder, "*/*/*/*.png"))))
    vimeo_images_list += list(glob(os.path.abspath(os.path.join(vimeo_folder, "*/*/*/*.jpg"))))

    train_images_list, val_images_list = train_test_split(
        vimeo_images_list,
        test_size=0.3,
        shuffle=True,
        random_state=2000
    )

    data_dict = {
        "train": train_images_list,
        "val": val_images_list
    }
    # Save data dictionary concludes train_images_path, val_images_path
    json_data_file = os.path.abspath(os.path.join(vimeo_data_folder, "data.json"))
    with open(json_data_file, "w") as f:
        json.dump(data_dict, f, indent=4)

    print(
        "Created train-val data json file!!! "
        f"\n{len(train_images_list)} train images"
        f"\n{len(val_images_list)} val images"
    )
    print("-" * 50)

    return


if __name__ == "__main__":
    create_kodak_json()
    create_vimeo_json()
