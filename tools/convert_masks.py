# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
from argparse import ArgumentParser
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np
from PIL import Image, ImageDraw
from skimage.draw import draw

from preprocess.convert_segmentation import load_instance
from utils.image_utils import read_image, read_image_np, save_image_np
from utils.io_utils import multi_glob_sorted

wnd_name = "ConvertMasksWindow"
inferred_color = np.array([[0, 0, 1]], dtype=np.float32)
mask_color = np.array([[0, 1, 0]], dtype=np.float32)
alpha = 0.4


def get_aabb(mask: np.ndarray) -> list[int]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return [0, 0, 0, 0]
    return [xs.min(), ys.min(), xs.max(), ys.max()]


def aabb_overlap(b0: list[int], b1: list[int]):
    """
    b0, b1: [x0, y0, x1, y1]
    """

    w = min(b0[2], b1[2]) - max(b0[0], b1[0])
    h = min(b0[3], b1[3]) - max(b0[1], b1[1])

    return max(w, 0) * max(h, 0)


class ConvertMasksTool:
    def __init__(self) -> None:
        parser = ArgumentParser()
        parser.add_argument("rgb", help="Path to images")
        parser.add_argument("--segmentation", help="Path to Detectron2 instance files (or zip)")
        parser.add_argument("--masks", help="Path to mask workspace")
        parser.add_argument("--idx", help="Start at index", type=int, default=0)
        args = parser.parse_args()

        ds_root = Path(args.rgb)

        # determine if first arg is root of dataset
        img_files = multi_glob_sorted(ds_root, ["*.png", "*.jpg"])
        if len(img_files) > 10:
            ds_root = ds_root.parent
        else:
            img_files = multi_glob_sorted(ds_root / "rgb_1x", ["*.png", "*.jpg"])

        if args.segmentation is None:
            args.segmentation = str(ds_root / "segmentation.zip")
        if args.masks is None:
            args.masks = str(ds_root / "masks")

        # load segmentation file list
        self.seg_root = Path(args.segmentation)
        self.seg_is_zip = args.segmentation.endswith(".zip")
        if self.seg_is_zip:
            zf = ZipFile(self.seg_root, mode="r")
            seg_files = sorted([f for f in zf.filelist if f.filename.endswith(".ckpt")], key=lambda zi: zi.filename)
            zf.close()
        else:
            seg_files = multi_glob_sorted(self.seg_root, ["*.ckpt"])
        assert len(img_files) == len(seg_files), f"Found {len(img_files)} images, but {len(seg_files)} segmentations"
        self.seg_files = seg_files

        # setup workspace
        workspace = Path(args.masks)
        workspace.mkdir(parents=True, exist_ok=True)

        mask_root = workspace / "mask"

        # determine mask sequence path (mask_folder)
        existing_folders = sorted([d for d in mask_root.glob("*") if os.path.isdir(d)])
        assert len(existing_folders) < 99, "You really need 99 masks??"

        if len(existing_folders) > 0:
            print("Found existing masks, choose which you'd like to modify:")
            for i in range(len(existing_folders)):
                print(f"{i:>2} - mask folder {existing_folders[i].name}")
            target = input("Choose one (defaults to new mask sequence): ").strip()
            if len(target) == 0:
                mask_folder = next((folder for i in range(100) if not os.path.isdir(folder := mask_root / f"{i:02d}")))
            else:
                mask_folder = existing_folders[int(target)]
        else:
            mask_folder = mask_root / "00"
        mask_folder.mkdir(parents=True, exist_ok=True)

        self.img_files = img_files
        self.mask_folder = mask_folder

        # state
        self.idx = args.idx
        self.dirty = False
        self.need_save_mask = False
        self.image = Image.fromarray(np.zeros([100, 100, 3], dtype=np.uint8))
        self.mask = np.zeros([100, 100], dtype=np.bool8)
        self.history = []
        self.is_inferred = False
        self.seg_idx = -1
        self.seg_boxes = []
        self.seg_masks = []

        # drawing
        self.drawing = False
        self.drawing_adding = True
        self.drawing_path = []

    def get_mask_file(self, idx: int):
        return self.mask_folder / f"{self.img_files[idx].stem}.png"

    def pop_history(self):
        if len(self.history) > 0:
            self.mask = self.history.pop()
            self.dirty = True

    @property
    def curr_mask_file(self):
        return self.get_mask_file(self.idx)

    def load_mask(self):
        # load seg masks
        idx = self.idx
        self.seg_idx = -1

        if self.seg_is_zip:
            zf = ZipFile(self.seg_root)
            zi = self.seg_files[idx]
            with zf.open(zi, "r") as f:
                seg_boxes, seg_masks = load_instance(f)
            zf.close()
        else:
            seg_boxes, seg_masks = load_instance(self.seg_files[idx])

        self.seg_boxes = seg_boxes
        self.seg_masks = seg_masks

        # set default
        self.need_save_mask = True
        self.mask = np.zeros([self.image.height, self.image.width], dtype=np.bool8)
        self.is_inferred = False
        mask_exists = False

        # if mask file already exists, use it
        if os.path.isfile(self.curr_mask_file):
            self.mask = read_image_np(self.curr_mask_file) > 0.5
            self.need_save_mask = False
            mask_exists = True

        # if previous mask is set, sort boxes by IOU
        prev_mask_file = self.get_mask_file(idx - 1)
        if idx == 0 or not os.path.isfile(prev_mask_file) or len(seg_masks) == 0:
            return

        prev_mask = read_image_np(prev_mask_file) > 0.5
        prev_box = get_aabb(prev_mask)
        prev_box_area = (prev_box[2] - prev_box[0]) * (prev_box[3] - prev_box[1])
        overlaps = np.array([aabb_overlap(b, prev_box) for b in seg_boxes], dtype=np.float32)
        areas = np.array([(b[2] - b[0]) * (b[3] - b[1]) for b in seg_boxes], dtype=np.float32)
        iou = overlaps / (areas + prev_box_area - overlaps)

        indices = np.argsort(-iou)
        seg_boxes = seg_boxes[indices]
        seg_masks = seg_masks[indices]

        self.seg_boxes = seg_boxes
        self.seg_masks = seg_masks

        # don't infer if exists or none overlaps
        if mask_exists or overlaps[indices[0]] < 1:
            return

        self.is_inferred = True
        self.mask = self.seg_masks[0]
        self.seg_idx = 0

    def load(self):
        print(f"Load mask {self.idx}")
        self.image = read_image(self.img_files[self.idx])
        self.load_mask()
        self.history = []

        # draw bounding boxes
        draw = ImageDraw.Draw(self.image)
        boxes = self.seg_boxes
        for i in range(len(boxes)):
            x, y = boxes[i][:2]
            draw.rectangle(boxes[i], outline=(255, 0, 0))
            draw.text((x + 2, y + 2), f"{i}", fill=(255, 0, 0))

        self.dirty = True

    def render(self):
        if not self.dirty:
            return

        img = np.array(self.image)[..., :3].astype(np.float32) / 255
        color = inferred_color if self.is_inferred else mask_color
        img[self.mask] = img[self.mask] * (1 - alpha) + color * alpha

        if self.drawing and len(self.drawing_path) > 1:
            color = [0, 1, 0] if self.drawing_adding else [1, 0, 0]
            color = np.array([color], dtype=img.dtype)

            for i in range(1, len(self.drawing_path)):
                x0, y0 = self.drawing_path[i]
                x1, y1 = self.drawing_path[i-1]
                rr, cc = draw.line(y0, x0, y1, x1)
                img[rr, cc] = color

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imshow(wnd_name, img)
        self.dirty = False

    def save(self):
        if self.need_save_mask:
            print(f"Save mask {self.idx} ({self.curr_mask_file.name})")
            save_image_np(self.curr_mask_file, self.mask)

    def move(self, delta):
        new_idx = min(len(self.img_files) - 1, max(0, self.idx + delta))
        if new_idx != self.idx:
            self.save()
            self.idx = new_idx
            self.load()

    def set_from_seg(self, seg_idx, append=False):
        if seg_idx < 0 or seg_idx >= len(self.seg_masks):
            print(f"Invalid seg index: {seg_idx}")
            return
        if append:
            self.set_mask(self.mask | self.seg_masks[seg_idx])
        else:
            self.set_mask(self.seg_masks[seg_idx])
        self.seg_idx = seg_idx

    def set_mask(self, mask):
        self.is_inferred = False
        self.history.append(self.mask)
        self.mask = mask
        self.dirty = True
        self.need_save_mask = True

    def handle_mouse_event(self, event, x, y, flags, param):
        # start and end
        if event == cv2.EVENT_LBUTTONDOWN:
            self.handle_mouse_move(x, y)
            self.drawing = True
            self.drawing_adding = True
            self.handle_mouse_move(x, y)
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.handle_mouse_move(x, y)
            self.drawing = True
            self.drawing_adding = False
        elif event == cv2.EVENT_LBUTTONUP or event == cv2.EVENT_RBUTTONUP:
            self.handle_mouse_move(x, y)
            self.finish_drawing()
            self.drawing = False
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.handle_mouse_move(x, y)

    def handle_mouse_move(self, x, y):
        self.drawing_path.append([x, y])
        self.dirty = True

    def finish_drawing(self):
        if len(self.drawing_path) == 0:
            return

        xs = [p[0] for p in self.drawing_path] + [self.drawing_path[-1][0]]
        ys = [p[1] for p in self.drawing_path] + [self.drawing_path[-1][1]]
        rr, cc = draw.polygon(ys, xs)
        mask = self.mask.copy()
        mask[rr, cc] = 1 if self.drawing_adding else 0
        self.set_mask(mask)

        self.drawing_path = []
        self.dirty = True

    def main(self):
        self.load()
        combine_mode = False

        cv2.namedWindow(wnd_name)
        cv2.setMouseCallback(wnd_name, self.handle_mouse_event)
        while True:
            self.render()

            key = cv2.waitKey(20) & 0xFFFF
            if key is None or key == 0xFFFF:
                continue
            if chr(key) == "q" or key == 27:  # ESC
                self.save()
                break
            elif key == 3:  # Right arrow
                self.move(1)
            elif key == 2:  # Left arrow
                self.move(-1)
            elif key == 0:  # Up arrow
                self.set_from_seg(max(self.seg_idx - 1, 0))
            elif key == 1:  # Down arrow
                self.set_from_seg(min(self.seg_idx + 1, len(self.seg_masks) - 1))
            elif key >= ord("0") and key <= ord("9"):  # Number
                self.set_from_seg(key - ord("0"), combine_mode)
                combine_mode = False
            elif chr(key) == "c":
                combine_mode = not combine_mode
                print(f"Combine mode is: {combine_mode}")
            elif chr(key) == "n":
                num = input("Index of box: ")
                try:
                    num = int(num.strip())
                except ValueError:
                    print("Invalid input")
                    continue
                self.set_from_seg(num, combine_mode)
                combine_mode = False
            elif chr(key) == "x":
                self.set_mask(np.zeros_like(self.mask))
            elif chr(key) == "z":
                self.pop_history()
            else:
                print("Unhandled key", key)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    tool = ConvertMasksTool()
    tool.main()
