from PIL import Image
import torch
import cv2
import numpy as np
from transformers import OwlViTProcessor
from owl_image_guided import (
    ImageGuidedOwlVit,
    post_process_image_guided_detection,
)
import os


def format(query_image_fpath, box):
    image = cv2.imread(query_image_fpath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    stencil = np.zeros(image.shape).astype(image.dtype)
    x1 = box[0] / image.shape[1]
    y1 = box[1] / image.shape[0]
    x2 = box[2] / image.shape[1]
    y2 = box[3] / image.shape[0]

    cont = np.array(
        [
            [x1 - 0.05, y1 - 0.05],
            [x2 + 0.05, y1 - 0.05],
            [x2 + 0.05, y2 + 0.05],
            [x1 - 0.05, y2 + 0.05],
        ]
    )
    cont = (cont * image.shape[:-1][::-1]).astype(np.int32)
    # cont[:, 0] -= 100
    # cont[:, 1] += 100
    color = [255, 255, 255]
    cv2.fillPoly(stencil, [cont], color)
    masked = cv2.bitwise_and(image, stencil)

    image_unmasked = Image.fromarray(image)
    image_masked = Image.fromarray(masked)
    return torch.tensor([x1, y1, x2, y2]).unsqueeze(0), image_unmasked, image_masked


def main(
    query_image_fpath, query_box, target_images, thresh=0.95, out_dir="detections"
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    query_box, query_unmasked, query_masked = format(query_image_fpath, query_box)
    query_unmasked.save("t.jpg")
    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    model = ImageGuidedOwlVit.from_pretrained("google/owlvit-base-patch32").to(device)
    model.eval()
    with torch.no_grad():
        for target_image in target_images:
            main_image = Image.open(target_image)
            main_image_cv2 = cv2.imread(target_image)
            # Only support single batchsize for now
            inputs = processor(
                query_images=query_masked, images=main_image, return_tensors="pt"
            ).to(device)

            import time

            _t = time.time()
            pred_logits, pred_boxes, sim = model.image_guided_detection(
                target_pixel_values=inputs["pixel_values"],
                query_pixel_values=inputs["query_pixel_values"],
                query_box=query_box,
            )
            results = post_process_image_guided_detection(
                pred_logits,
                pred_boxes,
                threshold=0.9,
                nms_threshold=0.3,
                target_image_size=torch.Tensor([main_image.size[::-1]]),
                sims=sim,
            ).pop()

            print(
                f"boxes: {len(results['boxes'])}\nscores: {[round(s, 2) for s in results['scores']]}"
            )

            if not len(results["boxes"]):
                continue

            for score, box in zip(results["scores"], results["boxes"]):
                main_image_cv2 = cv2.rectangle(
                    main_image_cv2,
                    [int(box[0]), int(box[1])],
                    [int(box[2]), int(box[3])],
                    (0, 255, 0),
                    10,
                )
            cv2.imwrite(f"{out_dir}/{os.path.basename(target_image)}", main_image_cv2)


if __name__ == "__main__":
    import glob

    box = [607.6289850050402, 264.2741935483871, 987.3709204889111, 358.33870967741933]

    # boxes = [
    #     [618.0805979082661, 278.2096774193548, 987.3709204889111, 347.88709677419354],
    #     [886.338662424395, 361.8225806451613, 534.4676946824596, 483.7580645161291],
    # ]
    query_impath = "datasets/marinesitu/01GGZ9TC2X60JVMBZ5QXXWSPG0.jpeg"
    target_images_dir = "datasets/marinesitu"
    target_images = glob.glob(f"{target_images_dir}/*")
    main(query_impath, box, target_images)

    # from owl_model_patched import OwlViTForObjectDetection
    # from patches import OwlViTPatched

    # model = OwlViTPatched.from_pretrained("google/owlvit-base-patch32")
    # print(model)
