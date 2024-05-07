import torch
import torch.nn as nn
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


class SCL(nn.Module):
    def __init__(self, strategy, temperature=0.1):
        super(SCL, self).__init__()
        self.strategy = strategy
        self.temperature = temperature
        self.knn_1 = KNeighborsClassifier(n_neighbors=1, metric='cosine')
        self.knn_2 = KNeighborsClassifier(n_neighbors=1, metric='cosine')

    @torch.no_grad()
    def relabel(self, real_features, real_labels, syn_features, syn_labels):
        z_real = real_features.detach().cpu().numpy()
        y_real = real_labels.detach().cpu().numpy()
        self.knn_1.fit(z_real[:, 0], y_real)
        self.knn_2.fit(z_real[:, 1], y_real)

        z_syn = syn_features.detach().cpu().numpy()
        y_syn = syn_labels.detach().cpu().numpy()
        output1 = self.knn_1.predict(z_syn[:, 0])
        output2 = self.knn_2.predict(z_syn[:, 1])
        
        index = ~((output1 == y_syn) & (output2 == y_syn))

        if self.strategy == "pos_neg":
            noise_labels = -1 * np.arange(1, np.sum(index) + 1)
            y_syn[index] = noise_labels
        else:
            y_syn[index] = -1
        
        return torch.from_numpy(y_syn).cuda()

    def forward(self, features, labels, is_syn, prototypes, prototypes_labels):
        real_features = features[is_syn == 0]
        real_labels = labels[is_syn == 0]
        syn_features = features[is_syn == 1]
        syn_labels = labels[is_syn == 1]

        real_features = torch.cat([real_features, prototypes], axis=0)
        real_labels = torch.cat([real_labels, prototypes_labels], axis=0)

        new_syn_labels = self.relabel(real_features, real_labels, syn_features, syn_labels)

        if self.strategy == 'drop':
            syn_features = syn_features[new_syn_labels != -1]
            new_syn_labels = new_syn_labels[new_syn_labels != -1]
        
        features = torch.cat([real_features, syn_features], axis=0)
        labels = torch.cat([real_labels, new_syn_labels], axis=0)

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

        if self.strategy == 'neg_only':
            # noise mask
            noise_mask = labels.flatten().repeat(anchor_count)
            noise_mask = torch.where(noise_mask == -1, 0, 1)
            
            loss = - mean_log_prob_pos * noise_mask
            loss = loss.view(anchor_count, batch_size).sum() / noise_mask.sum()
        else:
            loss = - mean_log_prob_pos
            loss = loss.view(anchor_count, batch_size).mean()

        return loss


if __name__ == '__main__':
    loss = SCL()
    x = torch.rand(4, 2, 3).float().cuda()
    label = torch.tensor([1, 0, 1, -1]).cuda()
    tmp = loss(x, label)