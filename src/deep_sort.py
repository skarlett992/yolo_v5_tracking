import numpy as np
import torch
import torchvision.transforms as transforms
import cv2
import logging
from src.tracker import Tracker
from torchreid import models

# детекция людей на фрагменте (изображении)
class Detection(object):
    """
    This class represents a bounding box detection in a single image.

    Parameters
    ----------
    tlwh : array_like
        Bounding box in format `(x, y, w, h)`.
    confidence : float
        Detector confidence score.
    feature : array_like
        A feature vector that describes the object contained in this image.

    Attributes
    ----------
    tlwh : ndarray
        Bounding box in format `(top left x, top left y, width, height)`.
    confidence : ndarray
        Detector confidence score.
    feature : ndarray | NoneType
        A feature vector that describes the object contained in this image.

    """

    def __init__(self, tlwh, confidence, feature):
        self.tlwh = np.asarray(tlwh, dtype=np.float)
        self.confidence = float(confidence)
        self.feature = np.asarray(feature, dtype=np.float32)

    def to_tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    def to_xyah(self):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

# экстрактор фич для результата детектора людей
class Extractor(object):
    def __init__(self, model_type, use_cuda=True):
        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        self.input_width = 128
        self.input_height = 256

        self.model = models.build_model(name=model_type, num_classes=1000)
        self.model.to(self.device)
        self.model.eval()

        logger = logging.getLogger("root.tracker")
        logger.info("Selected model type: {}".format(model_type))
        self.size = (self.input_width, self.input_height)
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def _preprocess(self, im_crops):
        def _resize(im, size):
            return cv2.resize(im.astype(np.float32) / 255., size)

        im_batch = torch.cat([self.norm(_resize(im, self.size)).unsqueeze(
            0) for im in im_crops], dim=0).float()
        return im_batch

    def __call__(self, im_crops):
        im_batch = self._preprocess(im_crops)
        with torch.no_grad():
            im_batch = im_batch.to(self.device)
            features = self.model(im_batch)
        return features.cpu().numpy()

# вспомогательная функция для расчета Евклидова расстояния (попарный квадрат расстояний)
# которое в свою очередь используется для расчета ближайшего соседа при трекинге
def _pdist(a, b):
    """Вычисление попарного квадрата расстояния между точками `a` и `b`.
    Параметры:
    ----------
    a : массив
        Матрица NxM с N значениями размера M.
    b : массив
        Матрица LxM с L значениями размера M.
    Returns
    -------
    ndarray
        Возвращает матрицу размером длины(a), длины(b), при этом элементы (i, j)
        содержат квадрат расстояния между `a[i]` и `b[j]`.
    """
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a2, b2 = np.square(a).sum(axis=1), np.square(b).sum(axis=1)
    r2 = -2. * np.dot(a, b.T) + a2[:, None] + b2[None, :]
    r2 = np.clip(r2, 0., float(np.inf))
    return r2

# Евклидово расстояние для трекинга по результатам детекции
def _nn_euclidean_distance(x, y):
    """ Helper function for nearest neighbor distance metric (Euclidean).
        Parameters
        ----------
        x : ndarray
            A matrix of N row-vectors (sample points).
        y : ndarray
            A matrix of M row-vectors (query points).
        Returns
        -------
        ndarray
            A vector of length M that contains for each entry in `y` the
            smallest Euclidean distance to a sample in `x`.
        """
    distances = _pdist(x, y)
    return np.maximum(0.0, distances.min(axis=0))

# вспомогательная функция расчета для косинусного расстояния
# которое в свою очередь используется для расчета ближайшего соседа при трекинге
def _cosine_distance(a, b, data_is_normalized=False):
    """Вычислить попарное косинусное расстояние между точками `a` и `b`.
    Параметры
    ----------
    a : массив
        Матрица NxM с N значениями размера M.
    b : массив
        Матрица LxM  с L samples размером M.
    data_is_normalized : Optional[bool]
        Если True, предполагает, что строки в a и b являются векторами единичной длины.
        В противном случае, a и b нормируются к длине 1.
    Returns
    -------
    ndarray
        Возвращает матрицу of size len(a), len(b) such that eleement (i, j)
        contains the squared distance between `a[i]` and `b[j]`.
    """
    if not data_is_normalized:
        a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)
        b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)
    return 1. - np.dot(a, b.T)

# косинусное расстояние для трекинга по результатам детекции
def _nn_cosine_distance(x, y):
    """ Helper function for nearest neighbor distance metric (cosine).
    Parameters
    ----------
    x : ndarray
        A matrix of N row-vectors (sample points).
    y : ndarray
        A matrix of M row-vectors (query points).
    Returns
    -------
    ndarray
        A vector of length M that contains for each entry in `y` the
        smallest cosine distance to a sample in `x`.
    """
    distances = _cosine_distance(x, y)
    return distances.min(axis=0)

# вычисление расстояния методом ближайшего соседа (евклидово либо косинусное) с целью трекинга результата детекции людей
class NearestNeighborDistanceMetric(object):
    """
        A nearest neighbor distance metric that, for each target, returns
        the closest distance to any sample that has been observed so far.
        Parameters
        ----------
        metric : str
            Either "euclidean" or "cosine".
        matching_threshold: float
            The matching threshold. Samples with larger distance are considered an
            invalid match.
        budget : Optional[int]
            If not None, fix samples per class to at most this number. Removes
            the oldest samples when the budget is reached.
        Attributes
        ----------
        samples : Dict[int -> List[ndarray]]
            A dictionary that maps from target identities to the list of samples
            that have been observed so far.
        """
    def __init__(self, metric, matching_threshold, budget=None):
        if metric == "euclidean":
            self._metric = _nn_euclidean_distance
        elif metric == "cosine":
            self._metric = _nn_cosine_distance
        else:
            raise ValueError(
                "Invalid metric; must be either 'euclidean' or 'cosine'")
        self.matching_threshold = matching_threshold
        self.budget = budget
        self.samples = {}

    def partial_fit(self, features, targets, active_targets):
        """Update the distance metric with new data.
        Parameters
        ----------
        features : ndarray
            An NxM matrix of N features of dimensionality M.
        targets : ndarray
            An integer array of associated target identities.
        active_targets : List[int]
            A list of targets that are currently present in the scene.
        """
        for feature, target in zip(features, targets):
            self.samples.setdefault(target, []).append(feature)
            if self.budget is not None:
                self.samples[target] = self.samples[target][-self.budget:]
        self.samples = {k: self.samples[k] for k in active_targets}

    def distance(self, features, targets):
        """Compute distance between features and targets.
        Parameters
        ----------
        features : ndarray
            An NxM matrix of N features of dimensionality M.
        targets : List[int]
            A list of targets to match the given `features` against.
        Returns
        -------
        ndarray
            Returns a cost matrix of shape len(targets), len(features), where
            element (i, j) contains the closest squared distance between
            `targets[i]` and `features[j]`.
        """
        cost_matrix = np.zeros((len(targets), len(features)))
        for i, target in enumerate(targets):
            cost_matrix[i, :] = self._metric(self.samples[target], features)
        return cost_matrix

# класс алгоритма deep sort
class DeepSort(object):
    def __init__(self, model_type, max_dist=0.2, max_iou_distance=0.7, max_age=70, n_init=3, nn_budget=100, use_cuda=True):

        self.extractor = Extractor(model_type, use_cuda=use_cuda)

        max_cosine_distance = max_dist
        metric = NearestNeighborDistanceMetric(
            "cosine", max_cosine_distance, nn_budget)
        self.tracker = Tracker(
            metric, max_iou_distance=max_iou_distance, max_age=max_age, n_init=n_init)

    def update(self, bbox_xywh, confidences, classes, ori_img, use_yolo_preds=True):
        self.height, self.width = ori_img.shape[:2]
        # generate detections
        features = self._get_features(bbox_xywh, ori_img)
        bbox_tlwh = self._xywh_to_tlwh(bbox_xywh)
        detections = [Detection(bbox_tlwh[i], conf, features[i]) for i, conf in enumerate(
            confidences)]

        # update tracker
        self.tracker.predict()
        self.tracker.update(detections, classes)

        # output bbox identities
        outputs = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            if use_yolo_preds:
                det = track.get_yolo_pred()
                x1, y1, x2, y2 = self._tlwh_to_xyxy(det.tlwh)
            else:
                box = track.to_tlwh()
                x1, y1, x2, y2 = self._tlwh_to_xyxy(box)
            track_id = track.track_id
            class_id = track.class_id
            outputs.append(np.array([x1, y1, x2, y2, track_id, class_id], dtype=np.int))
        if len(outputs) > 0:
            outputs = np.stack(outputs, axis=0)
        return outputs

    @staticmethod
    def _xywh_to_tlwh(bbox_xywh):
        if isinstance(bbox_xywh, np.ndarray):
            bbox_tlwh = bbox_xywh.copy()
        elif isinstance(bbox_xywh, torch.Tensor):
            bbox_tlwh = bbox_xywh.clone()
        bbox_tlwh[:, 0] = bbox_xywh[:, 0] - bbox_xywh[:, 2] / 2.
        bbox_tlwh[:, 1] = bbox_xywh[:, 1] - bbox_xywh[:, 3] / 2.
        return bbox_tlwh

    def _xywh_to_xyxy(self, bbox_xywh):
        x, y, w, h = bbox_xywh
        x1 = max(int(x - w / 2), 0)
        x2 = min(int(x + w / 2), self.width - 1)
        y1 = max(int(y - h / 2), 0)
        y2 = min(int(y + h / 2), self.height - 1)
        return x1, y1, x2, y2

    def _tlwh_to_xyxy(self, bbox_tlwh):
        x, y, w, h = bbox_tlwh
        x1 = max(int(x), 0)
        x2 = min(int(x+w), self.width - 1)
        y1 = max(int(y), 0)
        y2 = min(int(y+h), self.height - 1)
        return x1, y1, x2, y2

    def increment_ages(self):
        self.tracker.increment_ages()

    def _xyxy_to_tlwh(self, bbox_xyxy):
        x1, y1, x2, y2 = bbox_xyxy

        t = x1
        l = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        return t, l, w, h

    def _get_features(self, bbox_xywh, ori_img):
        im_crops = []
        for box in bbox_xywh:
            x1, y1, x2, y2 = self._xywh_to_xyxy(box)
            im = ori_img[y1:y2, x1:x2]
            im_crops.append(im)
        if im_crops:
            features = self.extractor(im_crops)
        else:
            features = np.array([])
        return features

