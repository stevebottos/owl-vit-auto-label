from PIL import Image
import torch
import cv2
from transformers import OwlViTProcessor
from owl_image_guided import ImageGuidedOwlVit, post_process
import os


def format_query(query_image_fpath, box):
    image = cv2.imread(query_image_fpath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    x1 = box[0] / image.shape[1]
    y1 = box[1] / image.shape[0]
    x2 = box[2] / image.shape[1]
    y2 = box[3] / image.shape[0]

    image_pil = Image.fromarray(image)
    return torch.tensor([x1, y1, x2, y2]).unsqueeze(0), image_pil, image


def draw_box_on_image(image, box, color=(0, 255, 0)):
    image = cv2.rectangle(
        image,
        [int(box[0]), int(box[1])],
        [int(box[2]), int(box[3])],
        color,
        10,
    )


@torch.no_grad()
def prepare_query(
    query_image_fpath, query_box, model, processor, device, use_classifier
):
    query_box, query_image, query_image_cv2 = format_query(query_image_fpath, query_box)
    query_image_processed = processor(query_images=query_image, return_tensors="pt")
    query_image_processed = query_image_processed["query_pixel_values"].to(device)

    with torch.no_grad():
        query_embeds, best_query_box, clf = model.get_query_box_features(
            query_image_processed, query_box, use_classifier
        )

    # sanity check:
    _best_query_box = best_query_box.squeeze(0).cpu().numpy() * [
        *query_image_cv2.shape[:-1][::-1],
        *query_image_cv2.shape[:-1][::-1],
    ]
    draw_box_on_image(query_image_cv2, _best_query_box)
    cv2.imwrite("debug/query_detected_box.jpg", query_image_cv2)

    return query_embeds, clf


@torch.no_grad()
def detect(target_image_fpath, query_embeds, model, processor, device, threshold):
    main_image = Image.open(target_image_fpath)
    target_image = processor(images=main_image, return_tensors="pt")
    target_image = target_image["pixel_values"].to(device)
    logits, boxes, embeddings = model.detect(target_image, query_embeds)

    results = post_process(
        logits,
        boxes,
        embeddings,
        threshold=threshold,
        target_image_size=torch.Tensor([main_image.size[::-1]]),
    ).pop()

    return results


def main(
    query_image_fpath,
    query_box,
    target_images,
    threshold=0.6,
    clf_threshold=0.3,
    out_dir="detections",
):
    use_classifier = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    model = (
        ImageGuidedOwlVit.from_pretrained("google/owlvit-base-patch32")
        .to(device)
        .eval()
    )
    query_embeds, clf = prepare_query(
        query_image_fpath, query_box, model, processor, device, use_classifier
    )

    for target_image_fpath in target_images:
        results = detect(
            target_image_fpath, query_embeds, model, processor, device, threshold
        )

        print(results["scores"])
        scores = results["scores"]
        boxes = results["boxes"]
        embeddings = results["embeddings"]

        if use_classifier and len(boxes):
            scores = torch.tensor(clf.predict_proba(embeddings)[:, 1])
            boxes = boxes[torch.where(scores > clf_threshold)]
            scores = scores[torch.where(scores > clf_threshold)]

        print(
            os.path.join("detections", os.path.basename(target_image_fpath)),
            f"\nboxes: {len(boxes)}\n"
            f"scores: {[round(s, 2) for s in scores.tolist()]}",
        )

        main_image_cv2 = cv2.imread(target_image_fpath)
        for box in boxes:
            draw_box_on_image(main_image_cv2, box)

        if len(boxes):
            cv2.imwrite(
                f"{out_dir}/{os.path.basename(target_image_fpath)}", main_image_cv2
            )


if __name__ == "__main__":
    import glob

    # SAMPLES
    # This one has one eel
    # query_impath = "datasets/marinesitu/01GGZ1P9T79KSG4V14HSFY5JTD.jpeg"
    # box = [325.4354366179435, 389.69354838709677, 854.9838237147177, 501.1774193548388]
    # target_images_dir = "datasets/marinesitu"

    # This has an eel and multiple fish
    # query_impath = "datasets/marinesitu/01GGZ9TC2X60JVMBZ5QXXWSPG0.jpeg"
    # box = [607.6289850050402, 264.2741935483871, 987.3709204889111, 358.33870967741933]
    # target_images_dir = "datasets/marinesitu"

    # Circuitboards
    query_impath = "datasets/circuits/92ba8cbf-a258-44b6-8b19-1476784ce57a.jpeg"
    box = [440.9934597876764, 880.6870967741936, 886.8321694650957, 1064.2677419354839]
    target_images_dir = "datasets/circuits"

    main(query_impath, box, glob.glob(f"{target_images_dir}/*"))
