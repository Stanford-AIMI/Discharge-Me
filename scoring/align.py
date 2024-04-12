import torch
import torch.nn as nn
from alignscore import AlignScore

class AlignScorer(nn.Module):
    def __init__(self):
        super(AlignScorer, self).__init__()
        self.align_scorer = AlignScore(
            model='roberta-base', 
            device='cpu',
            batch_size=8, 
            ckpt_path='https://huggingface.co/yzha/AlignScore/resolve/main/AlignScore-base.ckpt', 
            evaluation_mode='nli_sp')

    def forward(self, refs, hyps):
        f = self.align_scorer.score(
            contexts=refs,
            claims=hyps,
        )
        return f


if __name__ == "__main__":
    x, y = AlignScorer()(
        hyps=[            
            "there are moderate bilateral pleural effusions with overlying atelectasis,  underlying consolidation not excluded. mild prominence of the interstitial  markings suggests mild pulmonary edema. the cardiac silhouette is mildly  enlarged. the mediastinal contours are unremarkable. there is no evidence of  pneumothorax.",
            "there are moderate bilateral pleural effusions with overlying atelectasis,  underlying consolidation not excluded. mild prominence of the interstitial  markings suggests mild pulmonary edema. the cardiac silhouette is mildly  enlarged. the mediastinal contours are unremarkable. there is no evidence of  pneumothorax.",
            "there are moderate bilateral pleural effusions with overlying atelectasis,  underlying consolidation not excluded. mild prominence of the interstitial  markings suggests mild pulmonary edema. the cardiac silhouette is mildly  enlarged. the mediastinal contours are unremarkable. there is no evidence of  pneumothorax.",
        ],
        refs=[
            "heart size is moderately enlarged. the mediastinal and hilar contours are unchanged. there is no pulmonary edema. small left pleural effusion is present. patchy opacities in the lung bases likely reflect atelectasis. no pneumothorax is seen. there are no acute osseous abnormalities.",
            "heart size is mildly enlarged. the mediastinal and hilar contours are normal. there is mild pulmonary edema. moderate bilateral pleural effusions are present, left greater than right. bibasilar airspace opacities likely reflect atelectasis. no pneumothorax is seen. there are no acute osseous abnormalities.",
            "heart size is mildly enlarged. the mediastinal and hilar contours are normal. there is mild pulmonary edema. moderate bilateral pleural effusions are present, left greater than right. bibasilar airspace opacities likely reflect atelectasis. no pneumothorax is seen. there are no acute osseous abnormalities.",
        ],
    )
    print(x)
    print(y)
