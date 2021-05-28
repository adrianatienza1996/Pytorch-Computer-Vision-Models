import torch
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def xy_to_cxcy(xy):
    return torch.cat([(xy[:, 2:] + xy[:, :2]) / 2,  # c_x, c_y
                      xy[:, 2:] - xy[:, :2]], 1)  # w, h


def cxcy_to_xy(cxcy):
    return torch.cat([cxcy[:, :2] - (cxcy[:, 2:] / 2),  # x_min, y_min
                      cxcy[:, :2] + (cxcy[:, 2:] / 2)], 1)  # x_max, y_max


def cxcy_to_gcxgcy(cxcy, priors_cxcy):
    return torch.cat([(cxcy[:, :2] - priors_cxcy[:, :2]) / (priors_cxcy[:, 2:] / 10),  # g_c_x, g_c_y
                      torch.log(cxcy[:, 2:] / priors_cxcy[:, 2:]) * 5], 1)  # g_w, g_h


def gcxgcy_to_cxcy(gcxgcy, priors_cxcy):
    return torch.cat([gcxgcy[:, :2] * priors_cxcy[:, 2:] / 10 + priors_cxcy[:, :2],  # c_x, c_y
                      torch.exp(gcxgcy[:, 2:] / 5) * priors_cxcy[:, 2:]], 1)  # w, h

def find_jaccard_overlap(set_1, set_2):

    lower_bounds = torch.max(set_1[:, :2].unsqueeze(1), set_2[:, :2].unsqueeze(0))  # (n1, n2, 2)
    upper_bounds = torch.min(set_1[:, 2:].unsqueeze(1), set_2[:, 2:].unsqueeze(0))  # (n1, n2, 2)
    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)  # (n1, n2, 2)
    intersection = intersection_dims[:, :, 0] * intersection_dims[:, :, 1]  # (n1, n2)
    # Find areas of each box in both sets
    areas_set_1 = (set_1[:, 2] - set_1[:, 0]) * (set_1[:, 3] - set_1[:, 1])
    areas_set_2 = (set_2[:, 2] - set_2[:, 0]) * (set_2[:, 3] - set_2[:, 1])

    union = areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection

    return intersection / union


class MultiboxLoss(nn.Module):
    def __init__(self, priors_cxcy, num_classes, threshold=0.5, neg_pos_ratio=3, alpha=1.):
        super(MultiboxLoss, self).__init__()
        self.num_classes = num_classes
        self.priors_cxcy = priors_cxcy
        self.n_priors = priors_cxcy.size(0)
        self.priors_xy = cxcy_to_xy(priors_cxcy)
        self.threshold = threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = alpha
        self.loc_loss_fn = nn.L1Loss().to(device)
        label_loss_fn = nn.CrossEntropyLoss(reduce=False) if num_classes > 1 else nn.BCELoss(reduce=False)
        self.label_loss_fn = label_loss_fn.to(device)

    def forward(self, pred_locs, pred_labels, locs, labels):
        # locs and priors_xy must be in xy form
        batch_size = pred_locs.size(0)

        batch_true_locs = torch.zeros((batch_size, self.n_priors, 4), dtype=torch.float).to(device)
        batch_true_labels = torch.zeros((batch_size, self.n_priors), dtype=torch.long).to(device)
        for k in range(batch_size):
            # Intersection over union between priors and true localizations
            iou = find_jaccard_overlap(self.priors_xy, locs[k])

            # Mapping every prior to one object
            prior_overlap, prior_obj = iou.max(axis=1)
            true_labels = torch.tensor([labels[i] for i in prior_obj]).to(device)

            # Ensuring that every object is represented in one prior
            _, rep_prior = iou.max(axis=0)

            for i, j in zip(rep_prior, labels):
                prior_overlap[i] = 1.
                true_labels[i] = j

            cxcy_locs = xy_to_cxcy(locs[k])
            gxgy_coords = torch.cat([cxcy_to_gcxgcy(cxcy_locs[i].unsqueeze(0), self.priors_cxcy).unsqueeze(0)
                                     for i in range(cxcy_locs.size(0))]).to(device)
            true_locs = torch.cat([gxgy_coords[i, j, :].unsqueeze(0)
                                   for i, j in zip(prior_obj, range(prior_obj.size(0)))]).to(device)

            # Assigning predictions with confidence < threshold to 0 (background)
            true_labels[prior_overlap < self.threshold] = 0
            if self.num_classes == 1:
                true_labels = true_labels.float()

            # Storing the locs and labels of the image
            batch_true_locs[k] = true_locs
            batch_true_labels[k] = true_labels

        pos_matches_idx = batch_true_labels != 0

        # Localization loss
        loc_loss = self.loc_loss_fn(pred_locs[pos_matches_idx], true_locs[pos_matches_idx])

        # Confidence Loss
        n_positives = pos_matches_idx.sum(dim=1)  # (N)
        n_hard_negatives = self.neg_pos_ratio * n_positives

        conf_loss_all = self.label_loss_fn(pred_labels.view(-1), true_labels.view(-1))
        conf_loss_all = conf_loss_all.view(batch_size, self.n_priors)

        # Positive Conf Loss
        conf_loss_pos = conf_loss_all[pos_matches_idx]

        # Hard Mining for Negative Conf Loss
        conf_loss_neg = conf_loss_all.clone()
        conf_loss_neg[pos_matches_idx] = 0.
        conf_loss_neg, _ = conf_loss_neg.sort(dim=1, descending=True)
        hardness_ranks = torch.LongTensor(range(8732)).unsqueeze(0).expand_as(conf_loss_neg).to(device)
        hard_negatives = hardness_ranks < n_hard_negatives.unsqueeze(1)
        conf_loss_hard_neg = conf_loss_neg[hard_negatives]

        conf_loss = (conf_loss_hard_neg.sum() + conf_loss_pos.sum()) / n_positives.sum().float()
        return conf_loss + loc_loss


def train_batch(images, boxes, labels, model, opt, loss_fn):
    opt.zero_grad()
    model.train()
    predicted_locs, predicted_labels = model(images)
    multi_box_loss = loss_fn(predicted_locs, predicted_labels, boxes, labels)
    multi_box_loss.backward()
    opt.step()
    return multi_box_loss.item()


