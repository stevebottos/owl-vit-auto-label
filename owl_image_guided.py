import torch
from transformers import OwlViTForObjectDetection
from transformers.models.owlvit.modeling_owlvit import (
    generalized_box_iou,
    box_iou,
    center_to_corners_format,
)
from torchvision.ops import nms
from torch.nn.functional import cosine_similarity


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

    def get_query_box_features(self, query_image, query_box):
        query_features, _ = self.image_embedder(pixel_values=query_image)
        query_features_reshaped = self._reshape_feature_map(query_features)
        _, class_embeddings = self.class_predictor(query_features_reshaped)
        pred_boxes = center_to_corners_format(
            self.box_predictor(query_features_reshaped, query_features)
        )

        # TODO: Support more queries/batches
        assert pred_boxes.shape[0] == 1
        assert class_embeddings.shape[0] == 1

        query_image = query_image.squeeze(0)
        pred_boxes = pred_boxes.squeeze(0)
        class_embeddings = class_embeddings.squeeze(0)
        query_box = query_box.to(pred_boxes.device)

        ious, _ = box_iou(query_box, pred_boxes)
        # If there are no overlapping boxes, fall back to generalized IoU
        if torch.all(ious[0] == 0.0):
            ious = generalized_box_iou(query_box, pred_boxes)

        original_method = True
        if original_method:
            iou_threshold = torch.max(ious) * 0.8
            selected_indices = (ious[0] >= iou_threshold).nonzero()
            selected_embeddings = class_embeddings[selected_indices].squeeze(1)

            # Due to the DETR style bipartite matching loss, only one embedding
            # feature for each object is "good" and the rest are "background." To find
            # the one "good" feature we use the heuristic that it should be dissimilar
            # to the mean embedding.
            print("CS", selected_embeddings.shape)
            mean_embedding = torch.mean(selected_embeddings, axis=0)
            mean_sim = torch.einsum("d,id->i", mean_embedding, selected_embeddings)
            best_box_index = selected_indices[torch.argmin(mean_sim)]
        else:
            selected_indices = torch.tensor([torch.argmax(ious)])
            best_box_index = torch.tensor([torch.argmax(ious)])

        return class_embeddings[best_box_index], pred_boxes[best_box_index]

    def image_guided_detection(self, target_pixel_values, query_embeds):
        # Compute target image features
        feature_map, _ = self.image_embedder(pixel_values=target_pixel_values)
        target_image_features = self._reshape_feature_map(feature_map)

        # Predict object classes [batch_size, num_patches, num_queries+1]
        pred_logits, class_embeds = self.class_predictor(
            image_feats=target_image_features, query_embeds=query_embeds
        )

        # Predict object boxes
        pred_boxes = self.box_predictor(target_image_features, feature_map)
        sims = cosine_similarity(query_embeds, class_embeds, dim=-1)
        return pred_logits, pred_boxes, sims


def post_process_image_guided_detection(
    logits,
    target_boxes,
    threshold=0.7,
    nms_threshold=0.3,
    target_image_size=None,
    sims=None,
):
    probs = torch.max(logits, dim=-1)
    scores = torch.sigmoid(probs.values)
    # If there are no scores confident enough, then pass
    if scores.max() < threshold:
        return [{"scores": [], "sims": [], "boxes": []}]

    # Convert to [x0, y0, x1, y1] format
    target_boxes = center_to_corners_format(target_boxes)
    # Apply non-maximum suppression (NMS)
    for idx in range(target_boxes.shape[0]):
        for i in torch.argsort(-scores[idx]):
            if not scores[idx][i]:
                continue
            ious = box_iou(target_boxes[idx][i, :].unsqueeze(0), target_boxes[idx])[0][
                0
            ]
            ious[i] = -1.0  # Mask self-IoU.
            scores[idx][ious > nms_threshold] = 0.0

    torch_nms = nms(target_boxes.squeeze(0), scores.squeeze(0), iou_threshold=0.3)
    print(torch_nms)
    exit()
    # Convert from relative [0, 1] to absolute [0, height] coordinates
    img_h, img_w = target_image_size.unbind(1)
    scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(target_boxes.device)
    target_boxes = target_boxes * scale_fct[:, None, :]

    # Compute box display alphas based on prediction scores
    results = []
    for idx in range(target_boxes.shape[0]):
        # Select scores for boxes matching the current query:
        _scores = scores[idx]
        _boxes = target_boxes[idx]
        _sims = sims[idx]
        _boxes = _boxes[torch.where(_scores > threshold)]
        _sims = _sims[torch.where(_scores > threshold)]
        _scores = _scores[torch.where(_scores > threshold)]
        results.append(
            {
                "scores": _scores.tolist(),
                "sims": _sims.tolist(),
                "boxes": _boxes.tolist(),
            }
        )

    return results


# def og_post_process_image_guided_detection(
#     logits, target_boxes, threshold=0.7, nms_threshold=0.3, target_sizes=None
# ):
#     if len(logits) != len(target_sizes):
#         raise ValueError(
#             "Make sure that you pass in as many target sizes as the batch dimension of the logits"
#         )
#     if target_sizes.shape[1] != 2:
#         raise ValueError(
#             "Each element of target_sizes must contain the size (h, w) of each image of the batch"
#         )

#     probs = torch.max(logits, dim=-1)
#     scores = torch.sigmoid(probs.values)

#     # Convert to [x0, y0, x1, y1] format
#     target_boxes = center_to_corners_format(target_boxes)

#     # Apply non-maximum suppression (NMS)
#     if nms_threshold < 1.0:
#         for idx in range(target_boxes.shape[0]):
#             for i in torch.argsort(-scores[idx]):
#                 if not scores[idx][i]:
#                     continue

#                 ious = box_iou(target_boxes[idx][i, :].unsqueeze(0), target_boxes[idx])[
#                     0
#                 ][0]
#                 ious[i] = -1.0  # Mask self-IoU.
#                 scores[idx][ious > nms_threshold] = 0.0

#     # Convert from relative [0, 1] to absolute [0, height] coordinates
#     img_h, img_w = target_sizes.unbind(1)
#     scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(target_boxes.device)
#     target_boxes = target_boxes * scale_fct[:, None, :]

#     # Compute box display alphas based on prediction scores
#     results = []
#     alphas = torch.zeros_like(scores)

#     for idx in range(target_boxes.shape[0]):
#         # Select scores for boxes matching the current query:
#         query_scores = scores[idx]
#         if not query_scores.nonzero().numel():
#             continue

#         # Scale box alpha such that the best box for each query has alpha 1.0 and the worst box has alpha 0.1.
#         # All other boxes will either belong to a different query, or will not be shown.
#         max_score = torch.max(query_scores) + 1e-6
#         query_alphas = (query_scores - (max_score * 0.1)) / (max_score * 0.9)
#         query_alphas[query_alphas < threshold] = 0.0
#         query_alphas = torch.clip(query_alphas, 0.0, 1.0)
#         alphas[idx] = query_alphas

#         mask = alphas[idx] > 0
#         box_scores = alphas[idx][mask]
#         boxes = target_boxes[idx][mask]
#         results.append(
#             {"scores": box_scores.tolist(), "labels": None, "boxes": boxes.tolist()}
#         )

#     return results
