"""
    Routines for machine learning model training.
"""

__all__ = ['TrainProcessController']

import os
import shutil
from typing import Callable


class TrainProcessController(object):
    """
    This controller does the following things:
    1. Saves metrics and train parameters to a file in tabular form.
    2. Saves model checkpoints during training.
    3. Saves the best model checkpoints separately.
    4. Removes irrelevant model checkpoints.
    5. Logs the best metric values in a separate file in tabular form.
    6. Allows you to train in several attempts, with different training parameters.

    Parameters
    ----------
    checkpoint_file_name_prefix : str, default 'model'
        Prefix for checkpoint file name.
    last_checkpoint_file_name_suffix : str or None, default 'last'
        Suffix for last checkpoint file name. If None, then it is not used.
    best_checkpoint_file_name_suffix : str or None, default None
        Suffix for best checkpoint file name.
    last_checkpoint_dir_path : str, default ''
        Directory path for saving the last checkpoint files.
    best_checkpoint_dir_path : str or None, default None
        Directory name for saving the best checkpoint files.
        If None then best_checkpoint_dir_path = last_checkpoint_dir_path.
    last_checkpoint_file_count : int, default 2
        Count of the last checkpoint files to store.
    best_checkpoint_file_count : int, default 2
        Count of the best checkpoint files to store.
    checkpoint_file_save_callback : function or None, default None
        Callback for real saving of checkpoint file.
    checkpoint_file_exts : tuple(str, ...), default ('.params',)
        List of checkpoint file extensions.
    save_interval : int, default 1
        Interval of checkpoint file saving.
    num_epochs : int, default -1
        Number of epochs to force save the last checkpoint (important if save_interval > 1).
    param_names : tuple(str, ...) or None, default None
        Names for metrics and train parameters.
    key_metric_idx : int, default 0
        Index of key metric.
    score_log_file_path : str or None, default None
        File path to score log file.
    score_log_attempt_value : int, default 1
        Number of current attempt (used for comparing training curves for various hyperparameters).
    best_map_log_file_path : str or None, default None
        File path to best map log file.
    """
    def __init__(self,
                 checkpoint_file_name_prefix: str = "model",
                 last_checkpoint_file_name_suffix: str | None = "last",
                 best_checkpoint_file_name_suffix: str | None = None,
                 last_checkpoint_dir_path: str = "",
                 best_checkpoint_dir_path: str | None = None,
                 last_checkpoint_file_count: int = 2,
                 best_checkpoint_file_count: int = 2,
                 checkpoint_file_save_callback: Callable | None = None,
                 checkpoint_file_exts: tuple[str, ...] = (".params",),
                 save_interval: int = 1,
                 num_epochs: int = -1,
                 param_names: tuple[str, ...] | None = None,
                 key_metric_idx: int = 0,
                 score_log_file_path: str | None = None,
                 score_log_attempt_value: int = 1,
                 best_map_log_file_path: str | None = None):

        if not os.path.exists(last_checkpoint_dir_path):
            os.makedirs(last_checkpoint_dir_path)
        if best_checkpoint_dir_path is None:
            best_checkpoint_dir_path = last_checkpoint_dir_path
            assert ((last_checkpoint_file_name_suffix != best_checkpoint_file_name_suffix) and
                    (not ((last_checkpoint_file_name_suffix is None) and
                          (best_checkpoint_file_name_suffix is None))))
        else:
            assert (last_checkpoint_dir_path != best_checkpoint_dir_path)
            if not os.path.exists(best_checkpoint_dir_path):
                os.makedirs(best_checkpoint_dir_path)

        self.last_checkpoints_prefix = self._create_checkpoint_file_path_full_prefix(
            checkpoint_dir_path=last_checkpoint_dir_path,
            checkpoint_file_name_prefix=checkpoint_file_name_prefix,
            checkpoint_file_name_suffix=last_checkpoint_file_name_suffix)
        self.best_checkpoints_prefix = self._create_checkpoint_file_path_full_prefix(
            checkpoint_dir_path=best_checkpoint_dir_path,
            checkpoint_file_name_prefix=checkpoint_file_name_prefix,
            checkpoint_file_name_suffix=best_checkpoint_file_name_suffix)

        assert (last_checkpoint_file_count >= 0)
        self.last_checkpoint_file_count = last_checkpoint_file_count

        assert (best_checkpoint_file_count >= 0)
        self.best_checkpoint_file_count = best_checkpoint_file_count

        self.checkpoint_file_save_callback = checkpoint_file_save_callback
        self.checkpoint_file_exts = checkpoint_file_exts

        assert (save_interval > 0)
        self.save_interval = save_interval

        assert (num_epochs > 0)
        self.num_epochs = num_epochs

        assert isinstance(param_names, list)
        self.param_names = param_names

        assert (key_metric_idx >= 0) and (key_metric_idx < len(param_names))
        self.key_metric_idx = key_metric_idx

        required_titles = ["Attempt", "Epoch"]

        if score_log_file_path is not None:
            self.score_log_file_exist = (os.path.exists(score_log_file_path) and
                                         os.path.getsize(score_log_file_path) > 0)
            self.score_log_file = open(score_log_file_path, "a")
            if not self.score_log_file_exist:
                titles = required_titles + self.param_names
                self.score_log_file.write("\t".join(titles))
                self.score_log_file.flush()
        else:
            self.score_log_file = None

        self.score_log_attempt_value = score_log_attempt_value

        if best_map_log_file_path is not None:
            self.best_map_log_file_exist = (os.path.exists(best_map_log_file_path) and
                                            os.path.getsize(best_map_log_file_path) > 0)
            self.best_map_log_file = open(best_map_log_file_path, "a")
            if not self.best_map_log_file_exist:
                titles = required_titles + [self.param_names[self.key_metric_idx]]
                self.best_map_log_file.write("\t".join(titles))
                self.best_map_log_file.flush()
        else:
            self.best_map_log_file = None

        self.best_eval_metric_value = None
        self.best_eval_metric_epoch = None
        self.last_checkpoint_params_file_stems = []
        self.best_checkpoint_params_file_stems = []

        self.can_save = (self.checkpoint_file_save_callback is not None)

    def __del__(self):
        """
        Releasing resources.
        """
        if self.score_log_file is not None:
            self.score_log_file.close()
        if self.best_map_log_file is not None:
            self.best_map_log_file.close()

    def update_epoch_and_callback(self,
                                  epoch1: int,
                                  params: tuple[float | int, ...],
                                  **kwargs):
        """
        Update state after training epoch and probably call checkpoint_file_save_callback.

        Parameters
        ----------
        epoch1 : int
            Processed epoch number (started from 1).
        params : tuple(float or int, ...)
            Values for metrics and train parameters.
        **kwargs
            Extra arguments for checkpoint_file_save_callback.
        """
        curr_key_metric_value = params[self.key_metric_idx]
        if self.can_save:
            last_checkpoint_params_file_stem = None
            if (epoch1 % self.save_interval == 0) or (epoch1 == self.num_epochs):
                last_checkpoint_params_file_stem = self._get_last_checkpoint_params_file_stem(
                    epoch=epoch1,
                    acc=curr_key_metric_value)
                self.checkpoint_file_save_callback(last_checkpoint_params_file_stem, **kwargs)

                self.last_checkpoint_params_file_stems.append(last_checkpoint_params_file_stem)
                if len(self.last_checkpoint_params_file_stems) > self.last_checkpoint_file_count:
                    removed_checkpoint_file_stem = self.last_checkpoint_params_file_stems[0]
                    for ext in self.checkpoint_file_exts:
                        removed_checkpoint_file_path = removed_checkpoint_file_stem + ext
                        if os.path.exists(removed_checkpoint_file_path):
                            os.remove(removed_checkpoint_file_path)
                    del self.last_checkpoint_params_file_stems[0]

            if (self.best_eval_metric_value is None) or (curr_key_metric_value < self.best_eval_metric_value):
                self.best_eval_metric_value = curr_key_metric_value
                self.best_eval_metric_epoch = epoch1
                best_checkpoint_params_file_stem = self._get_best_checkpoint_params_file_stem(
                    epoch=epoch1,
                    acc=curr_key_metric_value)

                if last_checkpoint_params_file_stem is not None:
                    for ext in self.checkpoint_file_exts:
                        last_checkpoint_params_file_path = last_checkpoint_params_file_stem + ext
                        best_checkpoint_params_file_path = best_checkpoint_params_file_stem + ext
                        assert (os.path.exists(last_checkpoint_params_file_path))
                        shutil.copy(
                            src=last_checkpoint_params_file_path,
                            dst=best_checkpoint_params_file_path)
                else:
                    self.checkpoint_file_save_callback(best_checkpoint_params_file_stem, **kwargs)

                self.best_checkpoint_params_file_stems.append(best_checkpoint_params_file_stem)
                if len(self.best_checkpoint_params_file_stems) > self.best_checkpoint_file_count:
                    removed_checkpoint_file_stem = self.best_checkpoint_params_file_stems[0]
                    for ext in self.checkpoint_file_exts:
                        removed_checkpoint_file_path = removed_checkpoint_file_stem + ext
                        if os.path.exists(removed_checkpoint_file_path):
                            os.remove(removed_checkpoint_file_path)
                    del self.best_checkpoint_params_file_stems[0]

                if self.best_map_log_file is not None:
                    self.best_map_log_file.write("\n{:02d}\t{:04d}\t{:.4f}".format(
                        self.score_log_attempt_value, epoch1, curr_key_metric_value))
                    self.best_map_log_file.flush()
        if self.score_log_file is not None:
            score_log_file_row = "\n" + "\t".join([str(self.score_log_attempt_value), str(epoch1)] +
                                                  list(map(lambda x: "{:.4f}".format(x), params)))
            self.score_log_file.write(score_log_file_row)
            self.score_log_file.flush()

    @staticmethod
    def _create_checkpoint_file_path_full_prefix(checkpoint_dir_path: str,
                                                 checkpoint_file_name_prefix: str,
                                                 checkpoint_file_name_suffix: str):
        """
        Create checkpoint file path with full prefix.

        Parameters
        ----------
        checkpoint_dir_path : str
            Directory for checkpoint saving.
        checkpoint_file_name_prefix : str
            Checkpoint file name prefix.
        checkpoint_file_name_suffix : str
            Checkpoint file name suffix.
        """
        checkpoint_file_name_full_prefix = checkpoint_file_name_prefix
        if checkpoint_file_name_suffix is not None:
            checkpoint_file_name_full_prefix += ("_" + checkpoint_file_name_suffix)
        return os.path.join(
            checkpoint_dir_path,
            checkpoint_file_name_full_prefix)

    @staticmethod
    def _get_checkpoint_params_file_stem(checkpoint_file_path_prefix: str,
                                         epoch: int,
                                         acc: float):
        """
        Create checkpoint file stem path.

        Parameters
        ----------
        checkpoint_file_path_prefix : str
            Directory for checkpoint saving.
        epoch : int
            Epoch number.
        acc : float
            Accuracy value.
        """
        return "{}_{:04d}_{:.4f}".format(checkpoint_file_path_prefix, epoch, acc)

    def _get_last_checkpoint_params_file_stem(self,
                                              epoch: int,
                                              acc: float):
        """
        Create checkpoint file stem path for the last checkpoint.

        Parameters
        ----------
        epoch : int
            Epoch number.
        acc : float
            Accuracy value.
        """
        return self._get_checkpoint_params_file_stem(self.last_checkpoints_prefix, epoch, acc)

    def _get_best_checkpoint_params_file_stem(self,
                                              epoch: int,
                                              acc: float):
        """
        Create checkpoint file stem path for the best checkpoint.

        Parameters
        ----------
        epoch : int
            Epoch number.
        acc : float
            Accuracy value.
        """
        return self._get_checkpoint_params_file_stem(self.best_checkpoints_prefix, epoch, acc)
