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
                 category_id: Optional[int] = None,
                 isthing: Optional[bool] = None,
                 score: Optional[float] = None,
                 hidden_state_seg: Optional[np.ndarray] = None,
                 hidden_state_clip: Optional[np.ndarray] = None):
                
        self.id = id
        self.category_ids = [category_id]
        self.scores = [score]
        self.isthing = isthing
        self.hidden_states_clip = np.array([hidden_state_clip])
        if hidden_state_seg is not None:
            self.hidden_states_seg = np.array([hidden_state_seg])
        else:
            self.hidden_states_seg = None
        self.poke_count = 0  # number of detections since last this object was last seen

    def poke(self) -> None:
        self.poke_count += 1

    def unpoke(self) -> None:
        self.poke_count = 0

    def merge(self, other) -> None:
        
        self.hidden_states_clip = np.vstack((self.hidden_states_clip, other.hidden_states_clip))
        # self.hidden_states_clip = (self.hidden_states_clip * len(self.scores) + other.hidden_states_clip * len(other.scores)) / (len(self.scores) + len(other.scores))
        if self.hidden_states_seg is not None:
            self.hidden_states_seg = np.vstack((self.hidden_states_seg, other.hidden_states_seg))
            # self.hidden_states_seg = (self.hidden_states_seg * len(self.scores) + other.hidden_states_seg * len(other.scores)) / (len(self.scores) + len(other.scores))
        self.category_ids.extend(other.category_ids)
        self.scores.extend(other.scores)
        # self.hidden_states_seg.extend(other.hidden_states_seg)
        # self.hidden_states_clip.extend(other.hidden_states_clip)

    def vote_category_id(self) -> Optional[int]:
        category_ids = [c for c in self.category_ids if c is not None]
        if len(category_ids) == 0:
            return None
        else:
            return int(stats.mode(self.category_ids, keepdims=False)[0])

    def vote_score(self) -> Optional[float]:
        scores = [c for c in self.scores if c is not None]
        if len(scores) == 0:
            return None
        else:
            return float(np.mean(scores))

    def vote_hidden_states_seg(self) -> Optional[np.ndarray]:
        if self.hidden_states_seg is None:
            return None
        else:
            return self.hidden_states_seg.tolist()
        # if len(hidden_states_seg) == 0:
        #     return None
        # else:
        #     return np.mean(np.array(hidden_states_seg), axis=0).tolist()

    def vote_hidden_states_clip(self) -> Optional[np.ndarray]:
        # hidden_states_clip = [c for c in self.hidden_states_clip if c is not None]
        if len(self.hidden_states_clip) == 0:
            return None
        else:
            return self.hidden_states_clip.tolist()

    def get_rgb(self) -> np.ndarray:
        # this is valid for panoptic segmentation-style id only (0~255**3)
        return id_to_rgb(self.id)

    def copy_meta_info(self, other) -> None:
        self.category_ids = other.category_ids
        self.scores = other.scores
        self.isthing = other.isthing
        self.hidden_states_seg = other.hidden_states_seg
        self.hidden_states_clip = other.hidden_states_clip

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return self.id == other.id

    def __repr__(self):
        return f'(ID: {self.id}, cat: {self.category_ids}, isthing: {self.isthing}, score: {self.scores}, hidden_states_seg: {self.hidden_states_seg}, hidden_states_clip: {self.hidden_states_clip})'
