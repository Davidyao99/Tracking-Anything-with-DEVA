from typing import Optional
import numpy as np
from scipy import stats
from deva.utils.pano_utils import id_to_rgb


class ObjectInfo:
    """
    Stores meta information for an object
    """
    def __init__(self,
                 id: int,
                 label: str = None,
                 isthing: Optional[bool] = None,
                 score: Optional[float] = None,):
                
        self.id = id
        if label is not None:
            self.labels = [label]
        else:
            self.labels = []
        self.scores = [score]
        self.isthing = isthing
        self.poke_count = 0  # number of detections since last this object was last seen

    def poke(self) -> None:
        self.poke_count += 1

    def unpoke(self) -> None:
        self.poke_count = 0

    def merge(self, other) -> None:
        
        self.labels.extend(other.labels)
        self.scores.extend(other.scores)

    def vote_labels(self) -> Optional[str]:
        return self.labels

    def vote_score(self) -> Optional[float]:
        scores = [float(c) for c in self.scores if c is not None]
        if len(scores) == 0:
            return None
        else:
            return scores

    def get_rgb(self) -> np.ndarray:
        # this is valid for panoptic segmentation-style id only (0~255**3)
        return id_to_rgb(self.id)

    def copy_meta_info(self, other) -> None:
        self.labels = other.labels
        self.scores = other.scores
        self.isthing = other.isthing

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return self.id == other.id

    def __repr__(self):
        return f'(ID: {self.id}, labels: {self.labels}, isthing: {self.isthing}, score: {self.scores})'
