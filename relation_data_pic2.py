import torch
from piq import * 
import cv2
import json
from torchvision import transforms
from tqdm import tqdm
from scipy.stats import spearmanr

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

fr_metrics = [psnr, ssim, multi_scale_ssim, information_weighted_ssim, 
              vif_p, fsim, srsim, gmsd, multi_scale_gmsd, vsi, dss,
              haarpsi, mdsi]

fr_metrics_loss = [ContentLoss, StyleLoss, LPIPS, PieAPP, DISTS]

nr_metrics = [total_variation, brisque]

# , CLIPIQA

db_metrics = [IS, FID, GS,KID,  MSID, PR]

# real_scores_quality = []
# real_scores_authenticity = []
# real_scores_correspondence = []
# predicted_scores = {}



def get_scores(img1, img2, results) :
    for metric in fr_metrics:
        # print(metric.__name__, metric(img1, img2))
        score = metric(img1, img2)
        if metric.__name__ not in results:
            results[metric.__name__] = []
        
        results[metric.__name__].append(score.item())
    
    for metric in fr_metrics_loss:
        loss = metric()
        score = loss(img1, img2)
        if metric.__name__ not in results:
            results[metric.__name__] = []
        results[metric.__name__].append(score.item())

    for metric in nr_metrics:
                
        score = metric(img2)
        if metric.__name__ not in results:
            results[metric.__name__] = []
        results[metric.__name__].append(score.item())

    return results

def main():
    sd_predicted_scores = {}
    sd_real_scores = {'Quality': [], 'Authenticity': [], 'Correspondence': []}
    mj_predicted_scores = {}
    mj_real_scores = {'Quality': [], 'Authenticity': [], 'Correspondence': []}
    anno_file = 'data/I2IQA/anno.json'
    with open(anno_file, 'r') as f:
        anno = json.load(f)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        # resize
        transforms.Resize((256, 256)),
    ])
    
    # test
    count = 5
    for ref_path in tqdm(anno.keys()):
        img1 = cv2.imread(ref_path)
        img1 = transform(img1).unsqueeze(0)
        for aigc_anno in anno[ref_path]:
            img2 = cv2.imread(aigc_anno['path'])
            img2 = transform(img2).unsqueeze(0)
            model = aigc_anno['path'].split('/')[-1].split('_')[0]
            if model == 'SD':
                model = 'Stable Diffusion v1.5'
                sd_predicted_scores = get_scores(img1, img2, sd_predicted_scores)
                sd_real_scores['Quality'].append(aigc_anno['quality'])
                sd_real_scores['Authenticity'].append(aigc_anno['authenticity'])
                sd_real_scores['Correspondence'].append(aigc_anno['correspondence'])

            elif model == 'MJ':
                model = 'Midjourney'
                mj_predicted_scores = get_scores(img1, img2, mj_predicted_scores)
                mj_real_scores['Quality'].append(aigc_anno['quality'])
                mj_real_scores['Authenticity'].append(aigc_anno['authenticity'])
                mj_real_scores['Correspondence'].append(aigc_anno['correspondence'])
            count = count - 1
        # if count <= 0:
        #     break
    pred_results = {'SD': sd_predicted_scores, 'MJ': mj_predicted_scores}
    real_results = {'SD': sd_real_scores, 'MJ': mj_real_scores}
    json.dump(pred_results, open('predicted_scores_test.json', 'w'))
    json.dump(real_results, open('real_scores_test.json', 'w'))
if __name__ == '__main__':
    main()