import torch
from transformers import OwlViTForObjectDetection
from transformers.models.owlvit.modeling_owlvit import (
    generalized_box_iou,
    box_iou,
    center_to_corners_format,
)
from torchvision.ops import nms
from torch.nn.functional import cosine_similarity
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


class ImageGuidedOwlVit(OwlViTForObjectDetection):
    def __init__(self, config):
        super().__init__(config)

    def _reshape_feature_map(self, feature_map):
        return torch.reshape(
            feature_map,
            (
                feature_map.shape[0],
                feature_map.shape[1] * feature_map.shape[2],
                feature_map.shape[3],
            ),
        )

    #
    # EXPERIMENTAL! Use predictions that we know are wrong to help during post processing
    #
    def train_classifier(
        self, query_features_reshaped, query_embedding, best_box_index, class_embeddings
    ):
        query_logits, _class_embeddings = self.class_predictor(
            image_feats=query_features_reshaped, query_embeds=query_embedding
        )
        _class_embeddings = _class_embeddings.squeeze(0).cpu().numpy()

        probs = torch.max(query_logits, dim=-1)
        scores = torch.sigmoid(probs.values).squeeze(0).cpu().numpy()
        labels = np.zeros(len(_class_embeddings))
        labels[best_box_index.item()] = 1
        thresh = 0.8  # Keep only very high scoring negative labels else the model just learns the negative label
        _class_embeddings = _class_embeddings[scores > thresh]
        labels = labels[scores > thresh]

        assert len(_class_embeddings) > 5

        _query_embedding = class_embeddings[best_box_index].cpu().numpy()

        _class_embeddings = np.vstack([_class_embeddings, _query_embedding])
        labels = np.append(labels, 1)
        clf = MLPClassifier(random_state=1, max_iter=500, verbose=True)
        clf.fit(_class_embeddings, labels)

        return clf

    def get_query_box_features(self, query_image, query_box, train_classifier):
        query_features, _ = self.image_embedder(pixel_values=query_image)
        query_features_reshaped = self._reshape_feature_map(query_features)
        _, class_embeddings = self.class_predictor(image_feats=query_features_reshaped)
        _pred_boxes = self.box_predictor(query_features_reshaped, query_features)
        pred_boxes = center_to_corners_format(_pred_boxes)

        # TODO: Support more queries/batches
        assert pred_boxes.shape[0] == 1
        assert class_embeddings.shape[0] == 1

        query_image = query_image.squeeze(0)
        pred_boxes = pred_boxes.squeeze(0)
        class_embeddings = class_embeddings.squeeze(0)
        query_box = query_box.to(pred_boxes.device)

        query_aspect_ratio = (query_box[:, 2] - query_box[:, 0]) / (
            query_box[:, 3] - query_box[:, 1]
        )
        pred_aspect_ratios = (pred_boxes[:, 2] - pred_boxes[:, 0]) / (
            pred_boxes[:, 3] - pred_boxes[:, 1]
        )

        mask = torch.isclose(pred_aspect_ratios, query_aspect_ratio, atol=1)

        # pred_boxes = pred_boxes[mask]
        # class_embeddings = class_embeddings[mask]

        ious = box_iou(query_box, pred_boxes)[0] * mask
        # If there are no overlapping boxes, fall back to generalized IoU
        if torch.all(ious[0] == 0.0):
            print("Using generalized iou.")
            ious = generalized_box_iou(query_box, pred_boxes)

        iou_threshold = torch.max(ious) * 0.8
        selected_indices = (ious[0] >= iou_threshold).nonzero()
        selected_embeddings = class_embeddings[selected_indices].squeeze(1)

        # Due to the DETR style bipartite matching loss, only one embedding
        # feature for each object is "good" and the rest are "background." To find
        # the one "good" feature we use the heuristic that it should be dissimilar
        # to the mean embedding.
        mean_embedding = torch.mean(selected_embeddings, axis=0)
        mean_sim = torch.einsum("d,id->i", mean_embedding, selected_embeddings)
        best_box_index = selected_indices[torch.argmin(mean_sim)]

        query_embedding = class_embeddings[best_box_index]
        query_box = pred_boxes[best_box_index]

        clf = None
        if train_classifier:
            clf = self.train_classifier(
                query_features_reshaped,
                query_embedding,
                best_box_index,
                class_embeddings,
            )
        return (
            query_embedding,
            pred_boxes[best_box_index],
            clf,
        )

    def detect(self, target_pixel_values, query_embeds):
        # Compute target image features
        target_features, _ = self.image_embedder(pixel_values=target_pixel_values)
        target_features_reshaped = self._reshape_feature_map(target_features)

        # Predict object classes [batch_size, num_patches, num_queries+1]
        pred_logits, class_embeds = self.class_predictor(
            image_feats=target_features_reshaped, query_embeds=query_embeds
        )

        # Predict object boxes
        pred_boxes = self.box_predictor(target_features_reshaped, target_features)
        return pred_logits, pred_boxes, class_embeds


def post_process(
    logits,
    target_boxes,
    embeddings,
    threshold=0.7,
    iou_threshold=0.3,
    target_image_size=None,
):
    probs = torch.max(logits, dim=-1)
    scores = torch.sigmoid(probs.values).squeeze(0)
    embeddings = embeddings.squeeze(0)

    # Convert from relative [0, 1] to absolute [0, height] coordinates
    img_h, img_w = target_image_size.unbind(1)
    scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(target_boxes.device)
    target_boxes = target_boxes * scale_fct[:, None, :]

    target_boxes = target_boxes.squeeze(0)
    target_boxes = center_to_corners_format(target_boxes)

    indices = nms(target_boxes, scores, iou_threshold=iou_threshold)
    scores = scores[indices]
    target_boxes = target_boxes[indices]
    embeddings = embeddings[indices]

    return [
        {
            "scores": scores[torch.where(scores > threshold)],
            "boxes": target_boxes[torch.where(scores > threshold)],
            "embeddings": embeddings[torch.where(scores > threshold)].cpu().numpy(),
        }
    ]


# Slow
# def post_process_old(
#     logits,
#     target_boxes,
#     embeddings,
#     threshold=0.7,
#     iou_threshold=0.3,
#     target_image_size=None,
# ):
#     probs = torch.max(logits, dim=-1)
#     scores = torch.sigmoid(probs.values)

#     if scores.max() < threshold:
#         return [{"scores": [], "embeddings": [], "boxes": []}]

#     # Convert to [x0, y0, x1, y1] format
#     target_boxes = center_to_corners_format(target_boxes)
#     # Apply non-maximum suppression (NMS)
#     for idx in range(target_boxes.shape[0]):
#         for i in torch.argsort(-scores[idx]):
#             if not scores[idx][i]:
#                 continue
#             ious = box_iou(target_boxes[idx][i, :].unsqueeze(0), target_boxes[idx])[0][
#                 0
#             ]
#             ious[i] = -1.0  # Mask self-IoU.
#             scores[idx][ious > iou_threshold] = 0.0

#     # Convert from relative [0, 1] to absolute [0, height] coordinates
#     img_h, img_w = target_image_size.unbind(1)
#     scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(target_boxes.device)
#     target_boxes = target_boxes * scale_fct[:, None, :]

#     # Compute box display alphas based on prediction scores
#     results = []
#     for idx in range(target_boxes.shape[0]):
#         # Select scores for boxes matching the current query:
#         _scores = scores[idx]
#         _boxes = target_boxes[idx]

#         _boxes = _boxes[torch.where(_scores > threshold)]
#         _scores = _scores[torch.where(_scores > threshold)]
#         embeddings = embeddings[torch.where(scores > threshold)].cpu().numpy()
#         results.append({"scores": _scores, "boxes": _boxes, "embeddings": embeddings})

#     return results
