import json

def main():
    result_file = 'results.json'
    with open(result_file, 'r') as f:
        result = json.load(f)

    # sort result by quality score
    keys = ['Quality', 'Authenticity', 'Correspondence']
    # {"ssim": {
    #     "Quality": [
    #         -0.002350466203746594,
    #         0.9251526091931743
    #     ],
    #     "Authenticity": [
    #         0.046020268611280044,
    #         0.06571728737650016
    #     ],
    #     "Correspondence": [
    #         0.1068368510024745,
    #         1.848918081185598e-05
    #     ]
    # },}
    
    for key in keys:
        result_sorted = sorted(result.items(), key=lambda x: x[1][key][0], reverse=True)
        for i, (k, v) in enumerate(result_sorted):
            result[k][f'{key}_rank'] = 22 - (i + 1)

    for k in result.keys():
        for key in keys:
            result[k][key] = abs(result[k][key][0])
    # making pandas
            
    import pandas as pd
    df = pd.DataFrame(result)
    df = df.transpose()
    df = df.sort_values(by=['Quality_rank', 'Authenticity_rank', 'Correspondence_rank'], ascending=[False, False, False])
    df = df.transpose()
    # to csv
    df.to_csv('results_visualization.csv')

    
    breakpoint()
            
        
    



if __name__ == '__main__':
    main()