import os

import groundingdino.datasets.transforms as T
import numpy as np
import torch
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.inference import predict, predict_batch
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from huggingface_hub import hf_hub_download
from segment_anything import sam_model_registry
from segment_anything import SamPredictor

SAM_MODELS = {
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
}

CACHE_PATH = os.environ.get("TORCH_HOME", os.path.expanduser("~/.cache/torch/hub/checkpoints"))


def load_model_hf(repo_id, filename, ckpt_config_filename, device='cpu'):
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)

    args = SLConfig.fromfile(cache_config_file)
    model = build_model(args)
    args.device = device

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location='cpu')
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print(f"Model loaded from {cache_file} \n => {log}")
    model.eval()
    return model


def transform_image(image) -> torch.Tensor:
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    image_transformed, _ = transform(image, None)
    return image_transformed


class LangSAM():

    def __init__(self, sam_type="vit_h", ckpt_path=None):
        self.sam_type = sam_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.build_groundingdino()
        self.build_sam(ckpt_path)

    def build_sam(self, ckpt_path):
        if self.sam_type is None or ckpt_path is None:
            if self.sam_type is None:
                print("No sam type indicated. Using vit_h by default.")
                self.sam_type = "vit_h"
            checkpoint_url = SAM_MODELS[self.sam_type]
            try:
                sam = sam_model_registry[self.sam_type]()
                state_dict = torch.hub.load_state_dict_from_url(checkpoint_url)
                sam.load_state_dict(state_dict, strict=True)
            except:
                raise ValueError(f"Problem loading SAM please make sure you have the right model type: {self.sam_type} \
                    and a working checkpoint: {checkpoint_url}. Recommend deleting the checkpoint and \
                    re-downloading it.")
            sam.to(device=self.device)
            self.sam = SamPredictor(sam)
        else:
            try:
                sam = sam_model_registry[self.sam_type](ckpt_path)
            except:
                raise ValueError(f"Problem loading SAM. Your model type: {self.sam_type} \
                should match your checkpoint path: {ckpt_path}. Recommend calling LangSAM \
                using matching model type AND checkpoint path")
            sam.to(device=self.device)
            self.sam = SamPredictor(sam)

    def build_groundingdino(self):
        ckpt_repo_id = "ShilongLiu/GroundingDINO"
        ckpt_filename = "groundingdino_swinb_cogcoor.pth"
        ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"
        self.groundingdino = load_model_hf(ckpt_repo_id, ckpt_filename, ckpt_config_filename)

    def predict_dino(self, image_pil, text_prompt, box_threshold, text_threshold):
        image_trans = transform_image(image_pil)
        boxes, logits, phrases = predict(model=self.groundingdino,
                                         image=image_trans,
                                         caption=text_prompt,
                                         box_threshold=box_threshold,
                                         text_threshold=text_threshold,
                                         device=self.device)
        W, H = image_pil.size
        boxes = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])

        return boxes, logits, phrases

    def predict_dino_batch(self, image_pils, text_prompts, box_threshold, text_threshold):
        image_trans = [transform_image(image_pil) for image_pil in image_pils]
        boxes, logits, phrases = predict_batch(model=self.groundingdino,
                                               images=image_trans,
                                               captions=text_prompts,
                                               box_threshold=box_threshold,
                                               text_threshold=text_threshold,
                                               device=self.device)
        W, H = image_pils[0].size
        boxes = [box_ops.box_cxcywh_to_xyxy(box) * torch.Tensor([W, H, W, H]) for box in boxes]

        return boxes, logits, phrases

    def predict_sam(self, image_pil, boxes):
        image_array = np.asarray(image_pil)
        self.sam.set_image(image_array)
        transformed_boxes = self.sam.transform.apply_boxes_torch(boxes, image_array.shape[:2])
        masks, _, _ = self.sam.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes.to(self.sam.device),
            multimask_output=False,
        )
        return masks.cpu()

    def predict_sam_batch(self, image_pils, boxes):
        images = [np.asarray(image_pil) for image_pil in image_pils]

        def prepare_image(image):
            if "RGB" != self.sam.model.image_format:
                image = image[..., ::-1]
            image = self.sam.transform.apply_image(image)   # (H, W, 3)
            image = torch.as_tensor(image, device=self.device)
            image = image.permute(2, 0, 1).contiguous()   # (3, H, W)
            return image

        def transform_boxes(image, boxes):
            transformed_boxes = self.sam.transform.apply_boxes_torch(boxes, image.shape[:2])
            return transformed_boxes

        batched_input = []
        for i in range(len(images)):
            batched_input.append({
                "image": prepare_image(images[i]).to(self.sam.device),
                "boxes": transform_boxes(images[i], boxes[i]).to(self.sam.device),
                "original_size": images[i].shape[:2],
            })
        batched_output = self.sam.model(batched_input, multimask_output=False)
        masks = [o["masks"].cpu() for o in batched_output]  # Each mask output is (1, 1, H, W)
        return masks

    def predict(self, image_pil, text_prompt, box_threshold=0.3, text_threshold=0.25):
        boxes, logits, phrases = self.predict_dino(image_pil, text_prompt, box_threshold, text_threshold)
        masks = torch.tensor([])
        if len(boxes) > 0:
            masks = self.predict_sam(image_pil, boxes)
            masks = masks.squeeze(1)
        return masks, boxes, phrases, logits

    def predict_batch(self, image_pil, text_prompt, box_threshold=0.3, text_threshold=0.25):
        """
        Args:
            image_pil: list of PIL images
            text_prompt: list of strings
            box_threshold: float
            text_threshold: float
        Returns:
            masks: list of tensors (one tensor per image; each tensor is (num_boxes, H, W))
            boxes: list of tensors (one tensor per image; each tensor is (num_boxes, 4))
            phrases: list of strings
            logits: list of floats
        """
        assert len(image_pil) == len(text_prompt), "Must have same number of images and text prompts"
        boxes, logits, phrases = self.predict_dino_batch(image_pil, text_prompt, box_threshold, text_threshold)

        # Gather the box and image elements with boxes
        batch_images = []
        batch_boxes = []
        for i in range(len(boxes)):
            if len(boxes[i]) > 0:
                batch_images += [image_pil[i]]
                batch_boxes += [boxes[i]]
        if len(batch_images) > 0:
            print(f"Predicting SAM on {len(batch_boxes)} images")
            batch_masks = self.predict_sam_batch(batch_images, batch_boxes)
            batch_masks = [m.squeeze(1) for m in batch_masks]

        # Masks will be a list of tensors
        masks = []
        for i in range(len(boxes)):
            if len(boxes[i]) > 0:
                masks += [batch_masks.pop(0)]
            else:
                masks += [torch.tensor([])]

        return masks, boxes, phrases, logits
