import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SigLIPLoss(nn.Module):
    def __init__(self, logit_scale_init=0.07, logit_bias_init=0.0):
        super(SigLIPLoss, self).__init__()
        # Learnable parameters for scaling and bias
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / logit_scale_init))
        self.logit_bias = nn.Parameter(torch.ones([]) * logit_bias_init)
        
    def forward(self, audio_embeds, motion_embeds):
        # Normalize feature vectors
        audio_embeds = audio_embeds / audio_embeds.norm(p=2, dim=-1, keepdim=True)
        motion_embeds = motion_embeds / motion_embeds.norm(p=2, dim=-1, keepdim=True)
        
        # Compute similarity matrix
        logits = torch.matmul(audio_embeds, motion_embeds.t())
        
        # Apply scaling and bias
        logits = logits * self.logit_scale.exp() + self.logit_bias
        
        # Create identity matrix for positive pairs
        batch_size = audio_embeds.size(0)
        eye = torch.eye(batch_size, device=logits.device)
        
        # m1_diag1 will be -1 for negative pairs and +1 for positive pairs
        m1_diag1 = -torch.ones_like(logits) + 2 * eye
        
        # Apply sigmoid loss
        loglik = torch.nn.functional.logsigmoid(m1_diag1 * logits)
        
        # Sum over all pairs for each sample
        nll = -torch.sum(loglik, dim=-1)
        
        # Return mean loss
        loss = nll.mean()
        
        return loss


class LocalSigLIPLoss(nn.Module):
    def __init__(self, logit_scale_init=0.07, logit_bias_init=0.0):
        super(LocalSigLIPLoss, self).__init__()
        # Learnable parameters for scaling and bias
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / logit_scale_init))
        self.logit_bias = nn.Parameter(torch.ones([]) * logit_bias_init)
        
    def forward(self, motion_feature, audio_feature):
        batch_size, T, _ = motion_feature.size()
        assert len(motion_feature.shape) == 3

        # Normalize feature vectors
        motion_feature = F.normalize(motion_feature, dim=2)
        audio_feature = F.normalize(audio_feature, dim=2)

        motion_to_audio_loss = 0
        audio_to_motion_loss = 0
        motion_to_audio_correct = 0
        audio_to_motion_correct = 0
        
        # Get current scale and bias values
        logit_scale = self.logit_scale.exp()
        logit_bias = self.logit_bias

        # First pass: motion to audio
        for t in range(T):
            motion_feature_t = motion_feature[:, t, :]  # (bs, c)

            # Positive pair range for motion
            start = max(0, t - 4)
            end = min(T, t + 4)
            positive_audio_feature = audio_feature[:, start:end, :]  # (bs, pos_range, c)

            # Negative pair range for motion
            left_end = start
            left_start = max(0, left_end - 4 * 3)
            right_start = end
            right_end = min(T, right_start + 4 * 3)
            negative_audio_feature = torch.cat(
                [audio_feature[:, left_start:left_end, :], audio_feature[:, right_start:right_end, :]],
                dim=1
            )  # (bs, neg_range, c)

            # Concatenate positive and negative samples
            combined_audio_feature = torch.cat([positive_audio_feature, negative_audio_feature], dim=1)  # (bs, pos_range + neg_range, c)

            # Compute similarity scores with SigLIP scaling and bias
            logits = torch.matmul(motion_feature_t.unsqueeze(1), combined_audio_feature.transpose(1, 2))  # (bs, 1, pos_range + neg_range)
            logits = logits * logit_scale + logit_bias
            logits = logits.squeeze(1)  # (bs, pos_range + neg_range)

            # Create target matrix for sigmoid loss
            pos_count = positive_audio_feature.size(1)
            target = torch.zeros_like(logits)
            target[:, :pos_count] = 1.0  # Mark positive pairs with 1
            
            # Compute SigLIP loss (sigmoid-based)
            m1_diag1 = 2 * target - 1  # Convert 0/1 to -1/+1
            loglik = F.logsigmoid(m1_diag1 * logits)
            loss_t = -torch.sum(loglik, dim=1).mean()
            motion_to_audio_loss += loss_t

            # Compute accuracy
            max_indices = torch.argmax(logits, dim=1)
            correct_mask = (max_indices < pos_count).float()  # Check if indices are within the range of positive samples
            motion_to_audio_correct += correct_mask.sum()

        # Second pass: audio to motion
        for t in range(T):
            audio_feature_t = audio_feature[:, t, :]  # (bs, c)

            # Positive pair range for audio
            start = max(0, t - 4)
            end = min(T, t + 4)
            positive_motion_feature = motion_feature[:, start:end, :]  # (bs, pos_range, c)

            # Negative pair range for audio
            left_end = start
            left_start = max(0, left_end - 4 * 3)
            right_start = end
            right_end = min(T, right_start + 4 * 3)
            negative_motion_feature = torch.cat(
                [motion_feature[:, left_start:left_end, :], motion_feature[:, right_start:right_end, :]],
                dim=1
            )  # (bs, neg_range, c)

            # Concatenate positive and negative samples
            combined_motion_feature = torch.cat([positive_motion_feature, negative_motion_feature], dim=1)  # (bs, pos_range + neg_range, c)

            # Compute similarity scores with SigLIP scaling and bias
            logits = torch.matmul(audio_feature_t.unsqueeze(1), combined_motion_feature.transpose(1, 2))  # (bs, 1, pos_range + neg_range)
            logits = logits * logit_scale + logit_bias
            logits = logits.squeeze(1)  # (bs, pos_range + neg_range)

            # Create target matrix for sigmoid loss
            pos_count = positive_motion_feature.size(1)
            target = torch.zeros_like(logits)
            target[:, :pos_count] = 1.0  # Mark positive pairs with 1
            
            # Compute SigLIP loss (sigmoid-based)
            m1_diag1 = 2 * target - 1  # Convert 0/1 to -1/+1
            loglik = F.logsigmoid(m1_diag1 * logits)
            loss_t = -torch.sum(loglik, dim=1).mean()
            audio_to_motion_loss += loss_t

            # Compute accuracy
            max_indices = torch.argmax(logits, dim=1)
            correct_mask = (max_indices < pos_count).float()  # Check if indices are within the range of positive samples
            audio_to_motion_correct += correct_mask.sum()

        # Average the two losses
        final_loss = (motion_to_audio_loss + audio_to_motion_loss) / (2 * T)

        # Compute final accuracy
        total_correct = (motion_to_audio_correct + audio_to_motion_correct) / (2 * T * batch_size)
        
        return final_loss, total_correct



class LocalContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(LocalContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, motion_feature, audio_feature, learned_temp=None):
        if learned_temp is not None:
            temperature = learned_temp
        else:
            temperature = self.temperature
        batch_size, T, _ = motion_feature.size()
        assert len(motion_feature.shape) == 3

        motion_feature = F.normalize(motion_feature, dim=2)
        audio_feature = F.normalize(audio_feature, dim=2)

        motion_to_audio_loss = 0
        audio_to_motion_loss = 0
        motion_to_audio_correct = 0
        audio_to_motion_correct = 0

        # First pass: motion to audio
        for t in range(T):
            motion_feature_t = motion_feature[:, t, :]  # (bs, c)

            # Positive pair range for motion
            start = max(0, t - 4)
            end = min(T, t + 4)
            positive_audio_feature = audio_feature[:, start:end, :]  # (bs, pos_range, c)

            # Negative pair range for motion
            left_end = start
            left_start = max(0, left_end - 4 * 3)
            right_start = end
            right_end = min(T, right_start + 4 * 3)
            negative_audio_feature = torch.cat(
                [audio_feature[:, left_start:left_end, :], audio_feature[:, right_start:right_end, :]],
                dim=1
            )  # (bs, neg_range, c)

            # Concatenate positive and negative samples
            combined_audio_feature = torch.cat([positive_audio_feature, negative_audio_feature], dim=1)  # (bs, pos_range + neg_range, c)

            # Compute similarity scores
            logits = torch.matmul(motion_feature_t.unsqueeze(1), combined_audio_feature.transpose(1, 2)) / temperature  # (bs, 1, pos_range + neg_range)
            logits = logits.squeeze(1)  # (bs, pos_range + neg_range)

            # Compute InfoNCE loss
            positive_scores = logits[:, :positive_audio_feature.size(1)]
            loss_t = -positive_scores.logsumexp(dim=1) + torch.logsumexp(logits, dim=1)
            motion_to_audio_loss += loss_t.mean()

            # Compute accuracy
            max_indices = torch.argmax(logits, dim=1)
            correct_mask = (max_indices < positive_audio_feature.size(1)).float()  # Check if indices are within the range of positive samples
            motion_to_audio_correct += correct_mask.sum()

        # Second pass: audio to motion
        for t in range(T):
            audio_feature_t = audio_feature[:, t, :]  # (bs, c)

            # Positive pair range for audio
            start = max(0, t - 4)
            end = min(T, t + 4)
            positive_motion_feature = motion_feature[:, start:end, :]  # (bs, pos_range, c)

            # Negative pair range for audio
            left_end = start
            left_start = max(0, left_end - 4 * 3)
            right_start = end
            right_end = min(T, right_start + 4 * 3)
            negative_motion_feature = torch.cat(
                [motion_feature[:, left_start:left_end, :], motion_feature[:, right_start:right_end, :]],
                dim=1
            )  # (bs, neg_range, c)

            # Concatenate positive and negative samples
            combined_motion_feature = torch.cat([positive_motion_feature, negative_motion_feature], dim=1)  # (bs, pos_range + neg_range, c)

            # Compute similarity scores
            logits = torch.matmul(audio_feature_t.unsqueeze(1), combined_motion_feature.transpose(1, 2)) / temperature  # (bs, 1, pos_range + neg_range)
            logits = logits.squeeze(1)  # (bs, pos_range + neg_range)

            # Compute InfoNCE loss
            positive_scores = logits[:, :positive_motion_feature.size(1)]
            loss_t = -positive_scores.logsumexp(dim=1) + torch.logsumexp(logits, dim=1)
            audio_to_motion_loss += loss_t.mean()

            # Compute accuracy
            max_indices = torch.argmax(logits, dim=1)
            correct_mask = (max_indices < positive_motion_feature.size(1)).float()  # Check if indices are within the range of positive samples
            audio_to_motion_correct += correct_mask.sum()


        # Average the two losses
        final_loss = (motion_to_audio_loss + audio_to_motion_loss) / (2 * T)

        # Compute final accuracy
        total_correct = (motion_to_audio_correct + audio_to_motion_correct) / (2 * T * batch_size)
        
        return final_loss, total_correct



class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, feature1, feature2, learned_temp=None):
        batch_size = feature1.size(0)
        assert len(feature1.shape) == 2
        if learned_temp is not None:
            temperature = learned_temp
        else:
            temperature = self.temperature
        # Normalize feature vectors
        feature1 = F.normalize(feature1, dim=1)
        feature2 = F.normalize(feature2, dim=1)
        # Compute similarity matrix between feature1 and feature2
        similarity_matrix = torch.matmul(feature1, feature2.t()) / temperature
        # Extract positive similarities (diagonal elements)
        positive_similarities = torch.diag(similarity_matrix)
        # Compute the denominator using logsumexp for numerical stability
        denominator = torch.logsumexp(similarity_matrix, dim=1)
        # Compute the InfoNCE loss
        loss = - (positive_similarities - denominator).mean()
        return loss



# usage: like tot_loss = hinge_loss(logits_real) + hinge_loss(-logits_fake)
def hinge_loss(logits: torch.Tensor):
    return (1 - logits).relu().mean()


def softplus_loss(logits: torch.Tensor):
    return F.softplus(-logits).mean()


def linear_loss(logits: torch.Tensor):
    return (-logits).mean()


def focal_l1_loss(
    pred, target, reduction='none',
    alpha=0.2, gamma=1.0, activate='sigmoid', residual=False, weight=None
):
    r"""Calculate Focal L1 loss.

    Delving into Deep Imbalanced Regression. In ICML, 2021.
    <https://arxiv.org/abs/2102.09554>

    Args:
        pred (torch.Tensor): The prediction with shape (N, \*).
        target (torch.Tensor): The regression target with shape (N, \*).
        alpha (float): A balanced form for Focal Loss. Defaults to 0.2.
        gamma (float): The gamma for calculating the modulating factor.
            Defaults to 1.0.
        activate (str): activate methods in Focal loss in {'sigmoid', 'tanh'}.
            Defaults to 'sigmoid'.
        residual (bool): Whether to use the original l1_loss, i.e., l1 + focal_l1.
            Defaults to False.
        weight (tensor): Sample-wise reweight of (N, \*) or element-wise
            reweight of (1, \*). Defaults to None.
        reduction (str): The method used to reduce the loss.

    Returns:
        torch.Tensor: The calculated loss
    """
    _loss = F.l1_loss(pred, target, reduction='none')
    if activate == 'tanh':
        loss = _loss * (torch.tanh(alpha * _loss)) ** gamma
    else:
        loss = _loss * (2. * torch.sigmoid(alpha * _loss) - 1.) ** gamma
    if residual:
        loss += _loss
    
    if weight is not None:
        loss *= weight.expand_as(loss)
    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    return loss