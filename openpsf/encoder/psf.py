import logging
import numpy as np
import scipy
import torch

from .utils import create_sink, mask_valid_area


class Psf(object):
    def __init__(self, ann_rescale, skeleton, min_size=None):
        self.ann_rescale = ann_rescale
        self.skeleton = skeleton
        self.min_size = min_size

        self.log = logging.getLogger(self.__class__.__name__)

    def __call__(self, anns,anns2, width_height_original):
        keypoint_sets, bg_mask, valid_area = self.ann_rescale(anns, width_height_original)
        keypoint_sets2, bg_mask2, valid_area2 = self.ann_rescale(anns2, width_height_original)
        ##  valid_area should be the same
        bg_mask = bg_mask*bg_mask2
        #bg_mask[bg_mask2>0] = 1

        min_size = self.min_size
        if min_size is None:
            if bg_mask.shape[1] >= 60:
                min_size = 3
            else:
                min_size = 2

        self.log.debug('valid area: %s, paf min size = %d', valid_area, min_size)
        n_fields = keypoint_sets.shape[1]
        f = PsfGenerator(min_size, n_fields)
        f.init_fields(bg_mask)
        f.fill(keypoint_sets,keypoint_sets2)
        return f.fields(valid_area)


class PsfGenerator(object):
    def __init__(self, min_size, n_fields, v_threshold=0, padding=10):
        self.min_size = min_size
        self.n_fields = n_fields
        self.v_threshold = v_threshold
        self.padding = padding

        self.intensities = None
        self.fields_reg1 = None
        self.fields_reg2 = None
        self.fields_scale = None
        self.fields_reg_l = None

    def init_fields(self, bg_mask):
        n_fields = self.n_fields
        field_w = bg_mask.shape[1] + 2 * self.padding
        field_h = bg_mask.shape[0] + 2 * self.padding
        self.intensities = np.zeros((n_fields + 1, field_h, field_w), dtype=np.float32)
        self.fields_reg1 = np.zeros((n_fields, 2, field_h, field_w), dtype=np.float32)
        self.fields_reg2 = np.zeros((n_fields, 2, field_h, field_w), dtype=np.float32)
        self.fields_scale = np.zeros((n_fields, field_h, field_w), dtype=np.float32)
        self.fields_reg_l = np.full((n_fields, field_h, field_w), np.inf, dtype=np.float32)

        # set background
        self.intensities[-1] = 1.0
        self.intensities[-1, self.padding:-self.padding, self.padding:-self.padding] = bg_mask
        self.intensities[-1] = scipy.ndimage.binary_erosion(self.intensities[-1],
                                                            iterations=int(self.min_size / 2.0) + 1,
                                                            border_value=1)

    def fill(self, keypoint_sets,keypoint_sets2):
        # TODO(sven): remove randomization now?
        random_indices = np.random.choice(keypoint_sets.shape[0],
                                          keypoint_sets.shape[0],
                                          replace=False)
        #print(random_indices)
        for keypoints,keypoints2 in zip(keypoint_sets[random_indices],keypoint_sets2[random_indices]):
            self.fill_keypoints(keypoints,keypoints2)

    def fill_keypoints(self, keypoints,keypoints2):
        visible = keypoints[:, 2] > 0
        if not np.any(visible):
            return
        area = (
            (np.max(keypoints[visible, 0]) - np.min(keypoints[visible, 0])) *
            (np.max(keypoints[visible, 1]) - np.min(keypoints[visible, 1]))
        )
        scale = np.sqrt(area)

        for i in range(self.n_fields):
            joint1 = keypoints[i]
            joint2 = keypoints2[i]
            if joint1[2] <= self.v_threshold or joint2[2] <= self.v_threshold:
                continue

            self.fill_association(i, joint1, joint2, scale)

    def fill_association(self, i, joint1, joint2, scale):
        # offset between joints
        offset = joint2[:2] - joint1[:2] ## joint distance
        offset_d = np.linalg.norm(offset)

        # dynamically create s
        s = max(self.min_size, int(offset_d * 0.2))
        # s = self.min_size
        sink = create_sink(s)
        s_offset = (s - 1.0) / 2.0

        # pixel coordinates of top-left joint pixel
        joint1ij = np.round(joint1[:2] - s_offset)
        joint2ij = np.round(joint2[:2] - s_offset)
        offsetij = joint2ij - joint1ij

        # set fields
        num = max(2, int(np.ceil(offset_d)))
        fmargin = min(0.4, (s_offset + 1) / (offset_d + np.spacing(1)))
        # fmargin = 0.0
        for f in np.linspace(fmargin, 1.0-fmargin, num=num):
            fij = np.round(joint1ij + f * offsetij) + self.padding
            fminx, fminy = int(fij[0]), int(fij[1])
            fmaxx, fmaxy = fminx + s, fminy + s
           # print(self.intensities.shape)
            if fminx < 0 or fmaxx > self.intensities.shape[2] or \
               fminy < 0 or fmaxy > self.intensities.shape[1]:
                continue
            fxy = (fij - self.padding) + s_offset

            # precise floating point offset of sinks
            joint1_offset = (joint1[:2] - fxy).reshape(2, 1, 1)
            joint2_offset = (joint2[:2] - fxy).reshape(2, 1, 1)

            # update intensity
            self.intensities[i, fminy:fmaxy, fminx:fmaxx] = 1.0

            # update background
            self.intensities[-1, fminy:fmaxy, fminx:fmaxx] = 0.0

            # update regressions
            sink1 = sink + joint1_offset
            sink2 = sink + joint2_offset
            sink_l = np.minimum(np.linalg.norm(sink1, axis=0),
                                np.linalg.norm(sink2, axis=0))
            mask = sink_l < self.fields_reg_l[i, fminy:fmaxy, fminx:fmaxx]
            self.fields_reg1[i, :, fminy:fmaxy, fminx:fmaxx][:, mask] = \
                sink1[:, mask]
            self.fields_reg2[i, :, fminy:fmaxy, fminx:fmaxx][:, mask] = \
                sink2[:, mask]
            self.fields_reg_l[i, fminy:fmaxy, fminx:fmaxx][mask] = sink_l[mask]

            # update scale
            self.fields_scale[i, fminy:fmaxy, fminx:fmaxx][mask] = scale

    def fields(self, valid_area):
        intensities = self.intensities[:, self.padding:-self.padding, self.padding:-self.padding]
        fields_reg1 = self.fields_reg1[:, :, self.padding:-self.padding, self.padding:-self.padding]
        fields_reg2 = self.fields_reg2[:, :, self.padding:-self.padding, self.padding:-self.padding]
        fields_scale = self.fields_scale[:, self.padding:-self.padding, self.padding:-self.padding]

        intensities = mask_valid_area(intensities, valid_area)
        #print(np.unique(fields_reg1))
        return (
            torch.from_numpy(intensities),
            torch.from_numpy(fields_reg1),
            torch.from_numpy(fields_reg2),
            torch.from_numpy(fields_scale),
        )
