import torch
from piq import * 
import cv2
import json
from torchvision import transforms
from tqdm import tqdm
from scipy.stats import spearmanr

fr_metrics = [psnr, ssim, multi_scale_ssim, information_weighted_ssim, 
              vif_p, fsim, srsim, gmsd, multi_scale_gmsd, vsi, dss,
              haarpsi, mdsi]

fr_metrics_loss = [ContentLoss, StyleLoss, LPIPS, PieAPP, DISTS]

nr_metrics = [total_variation, brisque, CLIPIQA]

db_metrics = [IS, FID, GS,KID,  MSID, PR]

real_scores_quality = []
real_scores_authenticity = []
real_scores_correspondence = []

predicted_scores = {}

def main():
    
    anno_file = 'data/I2IQA/anno.json'
    with open(anno_file, 'r') as f:
        anno = json.load(f)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        # resize
        transforms.Resize((256, 256)),
    ])
    for ref_path in tqdm(anno.keys()):
        img1 = cv2.imread(ref_path)
        img1 = transform(img1).unsqueeze(0)
        for aigc_anno in anno[ref_path]:
            img2 = cv2.imread(aigc_anno['path'])
            img2 = transform(img2).unsqueeze(0)
            for metric in fr_metrics:
                # print(metric.__name__, metric(img1, img2))
                score = metric(img1, img2)
                if metric.__name__ not in predicted_scores:
                    predicted_scores[metric.__name__] = []
                predicted_scores[metric.__name__].append(score)
            
            for metric in fr_metrics_loss:
                loss = metric()
                score = loss(img1, img2)
                if metric.__name__ not in predicted_scores:
                    predicted_scores[metric.__name__] = []
                predicted_scores[metric.__name__].append(score)

            real_scores_quality.append(aigc_anno['quality'])
            real_scores_authenticity.append(aigc_anno['authenticity'])
            real_scores_correspondence.append(aigc_anno['correspondence'])
            

    print("Computing SRCC:")
    results = {}
    for metric in fr_metrics:
        if metric.__name__ not in results:
            results[metric.__name__] = {}
        results[metric.__name__]['Quality'] = spearmanr(real_scores_quality, predicted_scores[metric.__name__])
        results[metric.__name__]['Authenticity'] = spearmanr(real_scores_authenticity, predicted_scores[metric.__name__])
        results[metric.__name__]['Correspondence'] = spearmanr(real_scores_correspondence, predicted_scores[metric.__name__])
        # print(metric.__name__, "===================")
        # print("Quality: ", metric.__name__, spearmanr(real_scores_quality, predicted_scores[metric.__name__]))
        # print("Authenticity: ", metric.__name__, spearmanr(real_scores_authenticity, predicted_scores[metric.__name__]))
        # print("Correspondence", metric.__name__, spearmanr(real_scores_correspondence, predicted_scores[metric.__name__]))
    json.dump(results, open('results.json', 'w'))
if __name__ == '__main__':
    main()