from os import path as osp
import pandas as pd


def main():
    anno_file = "data/I2IQA/I2IQA_annotation.csv"
    anno = pd.read_csv(anno_file)
    reference_dir = "data/I2IQA/Image_prompt"
    generated_dir = "data/I2IQA/Generated_image/All"

    json_index = {}

    for idx, row in anno.iterrows():
        ref = row["Image_prompt"]
        gen = row["Generated_image"]
        quality = row["MOS_q"]
        authenticity = row["MOS_a"]
        correspondence = row["MOS_c"]

        ref_path = osp.join(reference_dir, ref)
        gen_path = osp.join(generated_dir, gen)
        ref_obj = json_index.get(ref_path, [])
        ref_obj.append({'path': gen_path, 
                        'quality':quality, 
                        'authenticity':authenticity, 
                        'correspondence':correspondence, 
                        'overall': (quality + authenticity + correspondence)})
        json_index[ref_path] = ref_obj
    
    # dump
    # import json
    # with open("data/I2IQA/anno.json", "w") as f:
    #     json.dump(json_index, f)

    # find the min overall score file
    
    min_score = 0x3f3f3f3f
    min_score_file = None
    for key, value in json_index.items():
        for item in value:
            if item['overall'] < min_score:
                min_score = item['overall']
                min_score_file = item['path']
    print("min overall: ",min_score_file, min_score)

    # find the max overall score file
    max_score = 0
    max_score_file = None
    for key, value in json_index.items():
        for item in value:
            if item['overall'] > max_score:
                max_score = item['overall']
                max_score_file = item['path']

    print("max overall: ", max_score_file, max_score)

    # find the min quality score file
    min_score = 0x3f3f3f3f
    min_score_file = None
    for key, value in json_index.items():
        for item in value:
            if item['quality'] < min_score:
                min_score = item['quality']
                min_score_file = item['path']
    print("min quality: ",min_score_file, min_score)

    # find the max quality score file
    max_score = 0
    max_score_file = None
    for key, value in json_index.items():
        for item in value:
            if item['quality'] > max_score:
                max_score = item['quality']
                max_score_file = item['path']
    
    print("max quality: ", max_score_file, max_score)

    # find the min authenticity score file
    min_score = 0x3f3f3f3f
    min_score_file = None
    for key, value in json_index.items():
        for item in value:
            if item['authenticity'] < min_score:
                min_score = item['authenticity']
                min_score_file = item['path']
    print("min authenticity: ",min_score_file, min_score)

    # find the max authenticity score file
    max_score = 0
    max_score_file = None
    for key, value in json_index.items():
        for item in value:
            if item['authenticity'] > max_score:
                max_score = item['authenticity']
                max_score_file = item['path']

    print("max authenticity: ", max_score_file, max_score)

    # find the min correspondence score file
    min_score = 0x3f3f3f3f
    min_score_file = None
    for key, value in json_index.items():
        for item in value:
            if item['correspondence'] < min_score:
                min_score = item['correspondence']
                min_score_file = item['path']
    print("min correspondence: ",min_score_file, min_score)

    # find the max correspondence score file
    max_score = 0
    max_score_file = None
    for key, value in json_index.items():
        for item in value:
            if item['correspondence'] > max_score:
                max_score = item['correspondence']
                max_score_file = item['path']

    print("max correspondence: ", max_score_file, max_score)
    


if __name__ == "__main__":
    main()
