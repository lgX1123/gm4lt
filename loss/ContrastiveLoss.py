import torch
import torch.nn as nn


class SCL(nn.Module):
    def __init__(self, temperature=0.1):
        super(SCL, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().cuda()

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        anchor_feature = contrast_feature
        anchor_count = contrast_count

        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).cuda(),
            0
        )
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # noise mask
        noise_mask = labels.flatten().repeat(anchor_count)
        noise_mask = torch.where(noise_mask == -1, 0, 1)

        # loss = - mean_log_prob_pos
        # loss = loss.view(anchor_count, batch_size).mean()

        loss = - mean_log_prob_pos * noise_mask
        loss = loss.view(anchor_count, batch_size).sum() / noise_mask.sum()

        return loss
    

if __name__ == '__main__':
    loss = SCL()
    x = torch.rand(4, 2, 3).float().cuda()
    label = torch.tensor([1, 0, 1, -1]).cuda()
    tmp = loss(x, label)