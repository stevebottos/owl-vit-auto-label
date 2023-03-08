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

    # cont = np.array(
    #     [
    #         [x1 - 0.05, y1 - 0.05],
    #         [x2 + 0.05, y1 - 0.05],
    #         [x2 + 0.05, y2 + 0.05],
    #         [x1 - 0.05, y2 + 0.05],
    #     ]
    # )
    # cont = (cont * image.shape[:-1][::-1]).astype(np.int32)
    # color = [255, 255, 255]
    # cv2.fillPoly(stencil, [cont], color)
    # masked = cv2.bitwise_and(image, stencil)

    image_pil = Image.fromarray(image)
    # image_masked = Image.fromarray(masked)
    return torch.tensor([x1, y1, x2, y2]).unsqueeze(0), image_pil, image


def draw_box_on_image(image, box):
    image = cv2.rectangle(
        image,
        [int(box[0]), int(box[1])],
        [int(box[2]), int(box[3])],
        (0, 255, 0),
        10,
    )
    return image


def main(
    query_image_fpath, query_box, target_images, thresh=0.95, out_dir="detections"
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    query_box, query_image, query_image_cv2 = format(query_image_fpath, query_box)

    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    model = ImageGuidedOwlVit.from_pretrained("google/owlvit-base-patch32").to(device)
    model.eval()
    with torch.no_grad():
        query_image = processor(query_images=query_image, return_tensors="pt")[
            "query_pixel_values"
        ].to(device)

        query_embeds, best_query_box = model.get_query_box_features(
            query_image, query_box
        )
        # To verify that the chosen box is legit
        # best_query_box = best_query_box.detach().cpu().numpy()[0]
        # print(best_query_box)
        # query_box = best_query_box * [
        #     *query_image_cv2.shape[:-1][::-1],
        #     *query_image_cv2.shape[:-1][::-1],
        # ]
        # print(query_box)
        # test_image = draw_box_on_image(query_image_cv2, query_box)
        # cv2.imwrite("test_image.jpg", test_image)
        # print(query_box)
        # exit()
        for target_image_path in target_images:
            main_image = Image.open(target_image_path)
            main_image_cv2 = cv2.imread(target_image_path)

            target_image = processor(images=main_image, return_tensors="pt")[
                "pixel_values"
            ].to(device)
            pred_logits, pred_boxes, sim = model.image_guided_detection(
                target_pixel_values=target_image, query_embeds=query_embeds
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
                f"boxes: {len(results['boxes'])}\nscores: {[round(s, 2) for s in results['scores']]}\nsims: {[round(s, 2) for s in results['sims']]}"
            )

            if not len(results["boxes"]):
                continue
            simscores = np.array(results["scores"]) * np.array(results["sims"])
            for simscore, box in zip(simscores, results["boxes"]):
                # if simscore < 0.8:
                #     continue
                draw_box_on_image(main_image_cv2, box)

            cv2.imwrite(
                f"{out_dir}/{os.path.basename(target_image_path)}", main_image_cv2
            )


if __name__ == "__main__":
    import glob

    # SAMPLES
    # This one has one eel
    # query_impath = "datasets/marinesitu/01GGZ1P9T79KSG4V14HSFY5JTD.jpeg"
    # box = [346.33866242439507, 403.6290322580645, 834.080597908266, 490.725806451613]

    # This has an eel and multiple fish
    query_impath = "datasets/marinesitu/01GGZ9TC2X60JVMBZ5QXXWSPG0.jpeg"
    box = [607.6289850050402, 264.2741935483871, 987.3709204889111, 358.33870967741933]

    target_images_dir = "datasets/marinesitu"
    target_images = glob.glob(f"{target_images_dir}/*")
    main(query_impath, box, target_images)

    # from owl_model_patched import OwlViTForObjectDetection
    # from patches import OwlViTPatched

    # model = OwlViTPatched.from_pretrained("google/owlvit-base-patch32")
    # print(model)
