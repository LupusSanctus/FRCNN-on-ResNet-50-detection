import io

import dvc.api as dvcapi
import torch

from detection.model.utils import Compose
from detection.model.utils import build_detector
from detection.model.utils import get_classes
from ..utils.config import Config
from ..utils.image import imread
from ..utils.parallel import collate, scatter


def init_detector(config, device='cuda:0'):
    """Initialize a detector from config file. DVC weights loading.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.

    Returns:
        nn.Module: The constructed detector.
    """
    if isinstance(config, str):
        config = Config.fromfile(config)
    elif not isinstance(config, Config):
        raise TypeError('config must be a filename or Config object, '
                        'but got {}'.format(type(config)))
    config.model.pretrained = None
    model = build_detector(config.model, test_cfg=config.test_cfg)

    weights = dvcapi.read('model_weights/faster_rcnn_r50_c4_2x-6e4fdf4f.pth', remote='gsremote', mode="rb",
                          encoding=None)
    checkpoint = torch.load(io.BytesIO(weights))
    state_dict = checkpoint['state_dict']

    model.load_state_dict(state_dict, strict=False)

    if 'CLASSES' in checkpoint['meta']:
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = get_classes('coco')

    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model


class LoadImage(object):

    def __call__(self, results):
        if isinstance(results['img'], str):
            results['filename'] = results['img']
        else:
            results['filename'] = None
        img = imread(results['img'])
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results


def inference_detector(model, img):
    """Inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
            images.

    Returns:
        If imgs is a str, a generator will be returned, otherwise return the
        detection results directly.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)
    # prepare data
    data = dict(img=img)
    data = test_pipeline(data)
    data = scatter(collate([data], samples_per_gpu=1), [device])[0]
    # forward the model
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)

    return result
