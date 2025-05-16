from datasets import Dataset, DatasetDict, concatenate_datasets, Sequence
# from datasets.table import embed_table_storage
from PIL import Image
import os
import json
from huggingface_hub import login
from tqdm import tqdm

login("")  # Your token

json_path = os.path.join(os.getcwd(), "_datasets", "ChartX_annotation.json")
with open(json_path, encoding='utf-8') as f:
    data = json.load(f)

dataset_dir = os.path.join(os.getcwd(), "_datasets")


def get_chart_info_json(chart_name):
    for data_element in data:
        if data_element["imgname"] == chart_name:
            return data_element
    return None


def load_image(path):
    return Image.open(path)

def get_concat_h(img_list):

    height_for_each_img = img_list[0].height
    width_for_each_img = img_list[0].width
    imgs_cnt = len(img_list)

    dst = Image.new('RGB', (width_for_each_img * imgs_cnt, height_for_each_img))
    
    for idx, img in enumerate(img_list):
        dst.paste(img, (width_for_each_img * idx, 0))
    # dst.paste(im1, (0, 0))
    # dst.paste(im2, (im1.width, 0))
    return dst

def prepare_unified_dataset(task_type, json_file, base_dir, original_img_dir = None):

    data = []
    list_ds = []
    batch_size = 100

    with open(json_file, 'r') as f:
        list_data = json.load(f)

    for meta in tqdm(list_data):
        imgname = meta['imgname']
        print(meta)
        img_json_info = get_chart_info_json(imgname)
        
        example = {}

        if task_type == "robustness":

            image_list = []
            # image list
            for idx in range(5):
                initial_img_path = os.path.join(base_dir, "imgs_" + str(meta["cell_change_cnt"]) + "_cell_change_" + meta["attr_change_type"] + "_altered", imgname, str(idx) + "_initial_chart" + ".png")
                modified_img_path = os.path.join(base_dir, "imgs_" + str(meta["cell_change_cnt"]) + "_cell_change_" + meta["attr_change_type"] + "_altered", imgname, str(idx) + "_modified_chart" + ".png")
                image_list = image_list + [load_image(initial_img_path), load_image(modified_img_path)]

            variant = [str(meta["cell_change_cnt"]) + "_data_point", meta["attr_change_type"]]
            attr_variation_values = {k: v["attribute_value"] for k, v in meta["chart_pair_imgs_set"].items()}
            pair_diff = json.dumps({"cell_change_JSON": meta["cell_change_JSON"], "attr_variation_values": attr_variation_values})

            example = {
                "chart_name": imgname,
                "chart_type": img_json_info["chart_type"],
                # "task": task_type,
                "difference_type": variant,
                "pair_difference_ground_truth": pair_diff,
                "image_pairs": get_concat_h(image_list)
            }

        else:
            orig_img = None
            try:
                # original image not processed
                orig_img_path = os.path.join(original_img_dir, imgname + ".png")
                orig_img = load_image(orig_img_path)
            except:
                continue

                pair_diff = None

            if task_type == "cell_alignment":
                final_img_path = os.path.join(base_dir, str(meta["cell_change_cnt"]) + "_cell_edit", imgname + ".png")
                final_img = load_image(final_img_path)

                variant = [str(meta["cell_change_cnt"]) + "_data_point"]
                pair_diff = json.dumps(meta["cell_change_json"])

            elif task_type == "attr_alignment":  
                final_img_path = os.path.join(base_dir, meta["attr_change_type"] + "_altered", imgname + ".png")
                final_img = load_image(final_img_path)
                
                variant = [meta["attr_change_type"]]
                pair_diff = json.dumps(meta["attr_change_json"]) 

            example = {
                "chart_name": imgname,
                "chart_type": img_json_info["chart_type"],
                # "task": task_type,
                "difference_type": variant,
                "pair_difference_ground_truth": pair_diff,
                "image_pairs": get_concat_h([orig_img, final_img])
            }


        data.append(example)

    return Dataset.from_list(data)

    print(f"len of samples: {len(data)}")
    for i in range(0, len(data), batch_size):
        chunk = data[i:i+batch_size]
        list_ds.append(Dataset.from_list(chunk))
    print(f"start concatenating ..")
    final_dataset = concatenate_datasets(list_ds)
    return final_dataset


if __name__ == "__main__":
    original_img_dir = "dataset_dir/ChartX_imgs_from_code_run"

    cell_dataset = prepare_unified_dataset(
        task_type="cell_alignment",
        json_file="dataset_dir/ChartX_cell_edit/cell_edit_info.json",
        base_dir="dataset_dir/ChartX_cell_edit",
        original_img_dir=original_img_dir
    )

    attr_dataset = prepare_unified_dataset(
        task_type="attr_alignment",
        json_file="dataset_dir/ChartX_attr_edit/attribute_edit_info.json",
        base_dir="dataset_dir/ChartX_attr_edit",
        original_img_dir=original_img_dir
    )

    robustness_dataset = prepare_unified_dataset(
        task_type="robustness",
        json_file="dataset_dir/ChartX_robustness/robustness_info.json",
        base_dir="dataset_dir/ChartX_robustness",
    )

    dataset_dict = DatasetDict({
        "data_alignment": cell_dataset,
        "attribute_alignment": attr_dataset,
        "robustness": robustness_dataset
    })

    print("Start uploading...")
    dataset_dict.push_to_hub("aniruddhbansal2000/ChartAlignBench", max_shard_size="4GB")
    print("Finished uploading.")
