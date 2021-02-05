import math

from .paf import Paf
from .pif import Pif
from .psf import Psf
from .skeleton import Skeleton
from .utils import anns_to_bg_mask, anns_to_keypoint_sets
from ..data import COCO_PERSON_SKELETON, DENSER_COCO_PERSON_SKELETON, KINEMATIC_TREE_SKELETON


def cli(parser):
    group = parser.add_argument_group('encoders')
    group.add_argument('--pif-side-length', default=None, type=int,
                       help='side length of the PIF field')
    group.add_argument('--paf-min-size', default=None, type=int,
                       help='min side length of the PAF field')


def factory(args, io_scales):
    encoders = []
    headnames = args.headnets

    for head_name, io_scale in zip(headnames, io_scales):
        if head_name in ('pif', 'pif17', 'ppif', 'pifb', 'pifs'):
            encoders.append(Pif(AnnRescaler(io_scale), args.pif_side_length))
        # elif head_name == 'pcf':
        #     encoders.append(Pcf(input_output_scale))
        elif head_name in ('paf', 'paf19', 'pafs'):
            encoders.append(Paf(AnnRescaler(io_scale), COCO_PERSON_SKELETON,
                                min_size=args.paf_min_size))
        elif head_name in ('skeleton',):
            encoders.append(Skeleton(AnnRescaler(io_scale)))
        elif head_name in ('pafb',):
            encoders.append(Paf(AnnRescaler(io_scale), COCO_PERSON_SKELETON,
                                min_size=args.paf_min_size))
        elif head_name == 'paf16':
            encoders.append(Paf(AnnRescaler(io_scale), KINEMATIC_TREE_SKELETON,
                                min_size=args.paf_min_size))
        elif head_name in ('paf44',):
            encoders.append(Paf(AnnRescaler(io_scale), DENSER_COCO_PERSON_SKELETON,
                                min_size=args.paf_min_size))
        elif head_name in ('psf','psf_co'):
            encoders.append(Psf(AnnRescaler(io_scale), COCO_PERSON_SKELETON,
                                min_size=args.paf_min_size))
        elif head_name in ('pisf'):
            encoders.append(Pif(AnnRescaler(io_scale), args.pif_side_length))
            encoders.append(Pisf(AnnRescaler(io_scale), COCO_PERSON_SKELETON,
                                min_size=args.paf_min_size))
        else:
            raise Exception('unknown head to create an encoder: {}'.format(head_name))

    if headnames[-1] == 'skeleton' and len(headnames) == len(io_scales) + 1:
        encoders.append(Skeleton(AnnRescaler(1)))
    # if head_name[0] in ('pisf'):
    #     encoders[1],encoders[2] = encoders[2],encoders[1]
    return encoders


class AnnRescaler(object):
    def __init__(self, input_output_scale):
        self.input_output_scale = input_output_scale

    def __call__(self, anns, width_height_original):
        keypoint_sets = anns_to_keypoint_sets(anns)
        keypoint_sets[:, :, :2] /= self.input_output_scale

        # background mask
        bg_mask = anns_to_bg_mask(width_height_original, anns)
        bg_mask = bg_mask[::self.input_output_scale, ::self.input_output_scale]

        # valid area
        valid_area = None
        if anns and 'valid_area' in anns[0]:
            valid_area = anns[0]['valid_area']
            valid_area = (
                valid_area[0] / self.input_output_scale,
                valid_area[1] / self.input_output_scale,
                valid_area[2] / self.input_output_scale,
                valid_area[3] / self.input_output_scale,
            )
        return keypoint_sets, bg_mask, valid_area
