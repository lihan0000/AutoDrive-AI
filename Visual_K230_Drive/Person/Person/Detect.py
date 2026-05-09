from libs.PipeLine import PipeLine, ScopedTiming
from libs.AIBase import AIBase
from libs.AI2D import Ai2d
import os
import ujson
from media.media import *
from time import *
import nncase_runtime as nn
import ulab.numpy as np
import time
import utime
import image
import random
import gc
import sys
import aidemo


class ObjectDetectionApp(AIBase):
    def __init__(self, kmodel_path, labels, model_input_size, max_boxes_num, confidence_threshold=0.5,
                 nms_threshold=0.2, rgb888p_size=[224, 224], display_size=[1920, 1080], debug_mode=0):
        super().__init__(kmodel_path, model_input_size, rgb888p_size, debug_mode)
        self.kmodel_path = kmodel_path
        self.labels = labels
        self.model_input_size = model_input_size
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.max_boxes_num = max_boxes_num
        self.rgb888p_size = [ALIGN_UP(rgb888p_size[0], 16), rgb888p_size[1]]
        self.display_size = [ALIGN_UP(display_size[0], 16), display_size[1]]
        self.debug_mode = debug_mode
        self.color_four = [(255, 220, 20, 60), (255, 119, 11, 32), (255, 0, 0, 142), (255, 0, 0, 230),
                           (255, 106, 0, 228), (255, 0, 60, 100), (255, 0, 80, 100), (255, 0, 0, 70),
                           (255, 0, 0, 192), (255, 250, 170, 30), (255, 100, 170, 30), (255, 220, 220, 0),
                           (255, 175, 116, 175), (255, 250, 0, 30), (255, 165, 42, 42), (255, 255, 77, 255),
                           (255, 0, 226, 252), (255, 182, 182, 255), (255, 0, 82, 0), (255, 120, 166, 157)]

        self.x_factor = float(self.rgb888p_size[0]) / self.model_input_size[0]
        self.y_factor = float(self.rgb888p_size[1]) / self.model_input_size[1]
        self.ai2d = Ai2d(debug_mode)
        self.ai2d.set_ai2d_dtype(nn.ai2d_format.NCHW_FMT, nn.ai2d_format.NCHW_FMT, np.uint8, np.uint8)

    def config_preprocess(self, input_image_size=None):
        with ScopedTiming("set preprocess config", self.debug_mode > 0):
            ai2d_input_size = input_image_size if input_image_size else self.rgb888p_size
            self.ai2d.resize(nn.interp_method.tf_bilinear, nn.interp_mode.half_pixel)
            self.ai2d.build([1, 3, ai2d_input_size[1], ai2d_input_size[0]],
                            [1, 3, self.model_input_size[1], self.model_input_size[0]])

    def postprocess(self, results):
        with ScopedTiming("postprocess", self.debug_mode > 0):
            result = results[0]
            result = result.reshape((result.shape[0] * result.shape[1], result.shape[2]))
            output_data = result.transpose()
            boxes_ori = output_data[:, 0:4]
            scores_ori = output_data[:, 4:]
            confs_ori = np.max(scores_ori, axis=-1)
            inds_ori = np.argmax(scores_ori, axis=-1)
            boxes, scores, inds = [], [], []
            for i in range(len(boxes_ori)):
                if confs_ori[i] > confidence_threshold:
                    scores.append(confs_ori[i])
                    inds.append(inds_ori[i])
                    x = boxes_ori[i, 0]
                    y = boxes_ori[i, 1]
                    w = boxes_ori[i, 2]
                    h = boxes_ori[i, 3]
                    left = int((x - 0.5 * w) * self.x_factor)
                    top = int((y - 0.5 * h) * self.y_factor)
                    right = int((x + 0.5 * w) * self.x_factor)
                    bottom = int((y + 0.5 * h) * self.y_factor)
                    boxes.append([left, top, right, bottom])
            if len(boxes) == 0:
                return []
            boxes = np.array(boxes)
            scores = np.array(scores)
            inds = np.array(inds)
            keep = self.nms(boxes, scores, nms_threshold)
            dets = np.concatenate((boxes, scores.reshape((len(boxes), 1)), inds.reshape((len(boxes), 1))), axis=1)
            dets_out = []
            for keep_i in keep:
                dets_out.append(dets[keep_i])
            dets_out = np.array(dets_out)
            dets_out = dets_out[:self.max_boxes_num, :]
            return dets_out  # dets_out

    def draw_result(self, pl, dets):
        with ScopedTiming("display_draw", self.debug_mode > 0):
            if dets:
                pl.osd_img.clear()
                for det in dets:
                    x1, y1, x2, y2 = map(lambda x: int(round(x, 0)), det[:4])
                    x = x1 * self.display_size[0] // self.rgb888p_size[0]
                    y = y1 * self.display_size[1] // self.rgb888p_size[1]
                    w = (x2 - x1) * self.display_size[0] // self.rgb888p_size[0]
                    h = (y2 - y1) * self.display_size[1] // self.rgb888p_size[1]
                    pl.osd_img.draw_rectangle(x, y, w, h, color=self.get_color(int(det[5])), thickness=4)
                    pl.osd_img.draw_string_advanced(x, y - 50, 32,
                                                    " " + self.labels[int(det[5])] + " " + str(round(det[4], 2)),
                                                    color=self.get_color(int(det[5])))
            else:
                pl.osd_img.clear()

    def nms(self, boxes, scores, thresh):
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = np.argsort(scores, axis=0)[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            new_x1, new_y1, new_x2, new_y2, new_areas = [], [], [], [], []
            for order_i in order:
                new_x1.append(x1[order_i])
                new_x2.append(x2[order_i])
                new_y1.append(y1[order_i])
                new_y2.append(y2[order_i])
                new_areas.append(areas[order_i])
            new_x1 = np.array(new_x1)
            new_x2 = np.array(new_x2)
            new_y1 = np.array(new_y1)
            new_y2 = np.array(new_y2)
            xx1 = np.maximum(x1[i], new_x1)
            yy1 = np.maximum(y1[i], new_y1)
            xx2 = np.minimum(x2[i], new_x2)
            yy2 = np.minimum(y2[i], new_y2)
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            new_areas = np.array(new_areas)
            ovr = inter / (areas[i] + new_areas - inter)
            new_order = []
            for ovr_i, ind in enumerate(ovr):
                if ind < thresh:
                    new_order.append(order[ovr_i])
            order = np.array(new_order, dtype=np.uint8)
        return keep

    def get_color(self, x):
        idx = x % len(self.color_four)
        return self.color_four[idx]

if __name__ == "__main__":
    display_mode = "lcd"
    if display_mode == "hdmi":
        display_size = [1920, 1080]
    else:
        display_size = [800, 480]
    kmodel_path = "/sdcard/app/tests/kmodel/test_Person.kmodel"
    labels = ["Person"]
    confidence_threshold = 0.2
    nms_threshold = 0.2
    max_boxes_num = 50
    rgb888p_size = [640, 640]
    pl = PipeLine(rgb888p_size=rgb888p_size, display_size=display_size, display_mode=display_mode)
    pl.create()
    ob_det = ObjectDetectionApp(kmodel_path, labels=labels, model_input_size=[640, 640], max_boxes_num=max_boxes_num,
                                confidence_threshold=confidence_threshold, nms_threshold=nms_threshold,
                                rgb888p_size=rgb888p_size, display_size=display_size, debug_mode=0)
    ob_det.config_preprocess()
    clock = time.clock()
    try:
        while True:
            os.exitpoint()
            clock.tick()
            img = pl.get_frame()
            res = ob_det.run(img)
            ob_det.draw_result(pl, res)
            pl.show_image()
            gc.collect()
            if not res:
                print(0)
            else:
                print(1)

    except Exception as e:
        sys.print_exception(e)

    finally:
        ob_det.deinit()
        pl.destroy()

#######################################################################################################################
