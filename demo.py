import argparse
import os
import time
import cv2
import numpy as np
import torch
from fusion.detector import TrunkDetector, TrunkTracker, OBBDetector, Detection, OBB
from fusion.formats import DebugImage, MaskImage, DetectionFormat


class SegmentationOverlay(DetectionFormat):
    """Overlays colored segmentation masks on the original image."""

    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.colors = {}

    def get_output_size(self, input_size):
        return input_size

    def generate(self, image, detections: dict[int, Detection]):
        overlay = image.copy()
        for det_id, det in detections.items():
            color = self._get_color(det_id)
            for component in det.components.values():
                if component.seg is not None:
                    contour = np.array(component.seg.contour.exterior.coords, dtype=np.int32)
                    cv2.fillPoly(overlay, [contour], color, lineType=cv2.LINE_AA)
        result = cv2.addWeighted(overlay, self.alpha, image, 1 - self.alpha, 0)
        # Draw black contours on top for visibility
        for det in detections.values():
            for component in det.components.values():
                if component.seg is not None:
                    contour = np.array(component.seg.contour.exterior.coords, dtype=np.int32)
                    cv2.polylines(result, [contour], isClosed=True, color=(0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        return result

    def _get_color(self, det_id):
        if det_id not in self.colors:
            self.colors[det_id] = tuple(int(c) for c in np.random.randint(50, 220, 3))
        return self.colors[det_id]


class SegmentationDetector(TrunkDetector):
    """Segmentation-only detector - skips OBB model, derives boxes from contours."""

    def __init__(self, seg_model_path):
        self.seg_model = self.load_model(self.DEFAULT_MODEL_SEG if seg_model_path is None else seg_model_path)
        self.obb_model = None

    def process_image(self, image, min_confidence=0.4) -> dict[int, Detection]:
        contours = self.detect_contours(image, min_confidence)

        detections = []
        for label in contours:
            for seg in contours[label]:
                obb = OBB(seg.obb(), seg.score)
                detections.append(Detection(label, Detection.Component(obb, seg, match_score=1.0)))

        return dict(enumerate(detections))


def parse_arguments():
    parser = argparse.ArgumentParser(description='demonstrator for tree-trunk detection and tracking '
                                                 'based on oriented object detection and instance segmentation')
    parser.add_argument('--input', type=str, required=True, help='input image directory or video file')
    parser.add_argument('--results_dir', type=str, required=True, help='directory for storing results')
    parser.add_argument('--obb_model', type=str, default=None,
                        help='path to model providing oriented bounding boxes (loads default model if not specified)')
    parser.add_argument('--seg_model', type=str, default=None,
                        help='path to model providing instance-segmentation contours '
                             '(loads default model if not specified)')
    parser.add_argument('--min_confidence', type=float, default=0.4,
                        help='minimum confidence for results provided by detection and segmentation models')
    parser.add_argument('--track', action='store_true', default=False, help='apply tracking across frames')
    parser.add_argument('--tracker_config', type=str, default='config/botsort_optimized.yaml',
                        help='path to tracker-configuration file')
    parser.add_argument('--clip_margin', type=float, default=0,
                        help='pixel margin from image border for identifying boundaries of clipped detections')
    parser.add_argument('--mask', action='store_true', default=False,
                        help='generate single-channel mask images instead of debug visualizations')
    parser.add_argument('--show_scores', action='store_true', default=False,
                        help='include scores and fused trunk obbs in visualization images')
    parser.add_argument('--seg_only', action='store_true', default=False,
                        help='use only segmentation model (skip OBB detection)')
    parser.add_argument('--obb_only', action='store_true', default=False,
                        help='use only OBB model (skip segmentation and fusion)')

    return parser.parse_args()


# process a single frame: run detection, generate output, collect timings
def process_frame(fusion, output_format, image, min_confidence):
    t0 = time.perf_counter()
    detections = fusion.process_image(image, min_confidence)
    detect_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    result = output_format.generate(image, detections)
    render_ms = (time.perf_counter() - t0) * 1000

    total_ms = detect_ms + render_ms
    return result, total_ms


# read a directory of images or a video file and apply detection and optionally tracking for each frame
def demo(obb_model_path, seg_model_path, input_path, results_dir, min_confidence, clip_margin, tracker_config,
         tracking, generate_masks, show_scores, seg_only=False, obb_only=False):
    os.makedirs(results_dir, exist_ok=True)

    # initialize detection/tracking and result format
    print('loading models...')
    if seg_only and obb_only:
        raise ValueError('cannot use --seg_only and --obb_only together')
    if seg_only:
        if tracking:
            print('Warning: tracking not supported with --seg_only, ignoring --track flag')
        fusion = SegmentationDetector(seg_model_path)
        output_format = MaskImage(clip_margin=clip_margin) if generate_masks else SegmentationOverlay()
    elif obb_only:
        if tracking:
            print('Warning: tracking not supported with --obb_only, ignoring --track flag')
        fusion = OBBDetector(obb_model_path)
        output_format = MaskImage(clip_margin=clip_margin) if generate_masks else DebugImage(
            clip_margin=clip_margin, draw_bounds=False, verbose=show_scores, draw_axis=True)
    elif tracking:
        fusion = TrunkTracker(obb_model_path, seg_model_path, tracker_config=tracker_config,
                              include_invalid=not generate_masks)
        output_format = MaskImage(clip_margin=clip_margin) if generate_masks else DebugImage(
            clip_margin=clip_margin, draw_bounds=False, verbose=show_scores, draw_axis=True)
    else:
        fusion = TrunkDetector(obb_model_path, seg_model_path)
        output_format = MaskImage(clip_margin=clip_margin) if generate_masks else DebugImage(
            clip_margin=clip_margin, draw_bounds=False, verbose=show_scores, draw_axis=True)

    cumulative_ms = 0.0
    frame_count = 0

    if os.path.isdir(input_path):
        # read input data from image files and generate result images
        for image_name in sorted(os.listdir(input_path)):
            image = cv2.imread(os.path.join(input_path, image_name))
            result, total_ms = process_frame(fusion, output_format, image, min_confidence)
            cv2.imwrite(os.path.join(results_dir, f'{os.path.splitext(image_name)[0]}.png'), result)
            torch.cuda.empty_cache()
            frame_count += 1
            cumulative_ms += total_ms
            avg_ms = cumulative_ms / frame_count
            print(f'\rimage {os.path.splitext(image_name)[0]}  '
                  f'total: {total_ms:.1f}ms  (avg: {avg_ms:.1f}ms / {1000.0 / avg_ms:.1f} fps)')
    else:
        # read input data from video file and generate result video
        cap_input = cv2.VideoCapture(input_path)
        output = cv2.VideoWriter(
            os.path.join(results_dir, f'result_{os.path.splitext(os.path.split(input_path)[-1])[0]}.avi'),
            cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), cap_input.get(cv2.CAP_PROP_FPS),
            output_format.get_output_size((int(cap_input.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                           int(cap_input.get(cv2.CAP_PROP_FRAME_HEIGHT)))))
        n_frames = int(cap_input.get(cv2.CAP_PROP_FRAME_COUNT))
        while cap_input.isOpened():
            ret, frame = cap_input.read()
            if not ret:
                break
            result, total_ms = process_frame(fusion, output_format, frame, min_confidence)
            output.write(result)
            torch.cuda.empty_cache()
            frame_count += 1
            cumulative_ms += total_ms
            avg_ms = cumulative_ms / frame_count
            print(f'frame {frame_count} / {n_frames}  '
                  f'total: {total_ms:.1f}ms  (avg: {avg_ms:.1f}ms / {1000.0 / avg_ms:.1f} fps)')
        cap_input.release()
        output.release()

    if frame_count > 0:
        avg_ms = cumulative_ms / frame_count
        print(f'\n--- Performance Summary ---')
        print(f'Frames processed: {frame_count}')
        print(f'Average per frame: {avg_ms:.1f}ms ({1000.0 / avg_ms:.1f} fps)')


def main():
    args = parse_arguments()
    demo(args.obb_model, args.seg_model, args.input, args.results_dir, args.min_confidence,
         args.clip_margin, args.tracker_config, args.track, args.mask, args.show_scores, args.seg_only,
         args.obb_only)


if __name__ == '__main__':
    main()
