from typing import Any, Dict, List, Optional, Tuple, Union

import logging
import shutil
import math
import bisect
import os
from glob import glob

logger = logging.getLogger(__name__)


class TopK(object):
    """
    Maintain top-k value using min heap.
    Currently, it allows duplicate
    """
    def __init__(self,
                 k: int) -> None:
        self.sorted_tuples: List[Tuple[Any, Any]] = []
        self.sorted_keys: List[Any] = []
        self.k = k

    def __str__(self):
        from pprint import pformat
        return pformat(self.sorted_tuples, indent=4)

    def __repr__(self):
        return super().__repr__() + "(\n" + self.__str__() + "\n)"

    def update(self, keyvalue: Tuple[Any, Any]) -> Tuple[bool, int, List[Tuple[int, int]]]:
        """
        O(log(n))
        """
        key, valu3 = keyvalue
        rank_changes = []  # (old_rank, new_rank)
        list_size = len(self.sorted_tuples)
        insert_location = bisect.bisect_right(self.sorted_keys, key)
        if list_size < self.k or insert_location > 0:
            # update list
            self.sorted_tuples.insert(insert_location, keyvalue)
            self.sorted_keys.insert(insert_location, key)
            if list_size == self.k:
                del self.sorted_tuples[0]
                del self.sorted_keys[0]

            # check rank changes
            if (list_size < self.k and insert_location > 0):
                for i in range(insert_location):
                    old_rank = list_size - i
                    new_rank = list_size + 1 - i
                    rank_changes.append((old_rank, new_rank))
            elif insert_location > 1:
                for i in range(insert_location - 1):
                    old_rank = list_size - 1 - i
                    new_rank = list_size - i
                    rank_changes.append((old_rank, new_rank))

            is_update = True
            kth_largest = list_size + 1 if insert_location == 0 \
                else list_size + 1 - insert_location
        else:
            is_update = False
            kth_largest = -1
        return is_update, kth_largest, rank_changes

    def kth_largest(self, k: int) -> Tuple[Any, Any]:
        """
        O(1)
        """
        assert k <= self.k
        return self.sorted_tuples[-k]


class CheckpointTracker(object):
    """
    This class implements the functionality for maintaing best checkpoints
    """
    def __init__(self,
                 checkpoint_path: str,
                 save_path_name: str = 'best_checkpoints',
                 max_to_keep: int = 1) -> None:
        self._src_path = checkpoint_path
        self._tgt_path = os.path.join(checkpoint_path, save_path_name)
        self._max_to_keep = max_to_keep
        self._tracker_state: TopK = TopK(max_to_keep)

    def update(self, score, step) -> bool:
        src_path, tgt_path = self._src_path, self._tgt_path
        is_update, kth_largest, rank_changes = self._tracker_state.update((score, step))
        os.makedirs(tgt_path, exist_ok=True)
        if is_update:
            logger.info(f"{kth_largest}-th best score so far. \
                        Copying weights to '{tgt_path}'.")
            src_fnames, num_splits = self._get_src_ckpt_name(step)
            tgt_fnames = self._get_tgt_ckpt_name(kth_largest, num_splits)

            # Copy old checkpoints
            for old_rank, new_rank in rank_changes:
                old_fnames = self._get_tgt_ckpt_name(old_rank, num_splits)
                new_fnames = self._get_tgt_ckpt_name(new_rank, num_splits)
                old_score_fname = os.path.join(tgt_path, f'{old_rank}th_info.txt')
                new_score_fname = os.path.join(tgt_path, f'{new_rank}th_info.txt')
                for old_fname, new_fname in zip(old_fnames, new_fnames):
                    shutil.copyfile(old_fname, new_fname)
                shutil.copyfile(old_score_fname, new_score_fname)

            # Copy new checkpoints
            for src_fname, tgt_fname in zip(src_fnames, tgt_fnames):
                shutil.copyfile(src_fname, tgt_fname)
            with open(os.path.join(tgt_path, f'{kth_largest}th_info.txt'), 'w') as fp:
                fp.write(f"Step: {step}, Score: {score}")
            return True
        else:
            return False

    def _get_src_ckpt_name(self, step):
        num_splits = int(glob(os.path.join(self._src_path, f'ckpt-{step}.data-00000-of-0000*'))[0].split('-')[-1])
        fname_templates =  [f'ckpt-{step}.index'] + \
            [f'ckpt-{step}.data-0000{i}-of-0000{num_splits}' for i in range(num_splits)]
        return map(lambda x: os.path.join(self._src_path, x), fname_templates), num_splits

    def _get_tgt_ckpt_name(self, kth_best, num_splits):
        fname_templates = [f'ckpt-{kth_best}th-best.index'] + \
            [f'ckpt-{kth_best}th-best.data-0000{i}-of-0000{num_splits}' for i in range(num_splits)]
        return map(lambda x: os.path.join(self._tgt_path, x), fname_templates)


def main():
    a = TopK(3)
    result = a.update((3, "a")); print(result, a) # 1, []
    result = a.update((2, "b")); print(result, a) # 2, []
    result = a.update((5, "c")); print(result, a) # 1, [(2, 3), (1,2 )]
    result = a.update((1, "d")); print(result, a) # -1, []
    result = a.update((7, "e")); print(result, a) # 1, [(2, 3), (1, 2)]
    result = a.update((6, "f")); print(result, a) # 2, [(2, 3)]
    result = a.update((9, "g")); print(result, a) # 1, [(2, 3), (1, 2)]
    result = a.update((11, "h")); print(result, a) # 1, [(2, 3), (1, 2)]
    result = a.update((8, "i")); print(result, a) # 3, []


if __name__ == '__main__':
    main()
