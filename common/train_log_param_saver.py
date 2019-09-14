import os
import shutil


class TrainLogParamSaver(object):
    """
    Train logger does the following:
    1. save several the last model checkpoints, for disaster recovery,
    2. save several the best model checkpoints, to prevent overfitting,
    3. save pure evaluation metric values to log-file for observer.

    Parameters:
    ----------
    checkpoint_file_name_prefix : str
        prefix for checkpoint file name (without parent dir)
    last_checkpoint_file_name_suffix : str or None
        suffix for last checkpoint file name
        if None then checkpoint_file_name_prefix is not modified
    best_checkpoint_file_name_suffix : str or None
        suffix for best checkpoint file name
    last_checkpoint_dir_path : str
        directory path for saving the last checkpoint files
    best_checkpoint_dir_path : str or None
        directory name for saving the best checkpoint files
        if None then best_checkpoint_dir_path = last_checkpoint_dir_path
    last_checkpoint_file_count : int
        count of the last checkpoint files
    best_checkpoint_file_count : int
        count of the best checkpoint files
    checkpoint_file_save_callback : function or None
        Callback for real saving of checkpoint file
    checkpoint_file_exts : tuple of str
        List of checkpoint file extensions
    save_interval : int
        Interval of checkpoint file saving
    num_epochs : int
        Number of epochs for saving last checkpoint if save_interval > 1
    bigger : list of bool
        Should be bigger for each value of evaluation metric values
    mask : list of bool or None
        evaluation metric values that should be taken into account
    score_log_file_path : str or None
        file path to score log file
    score_log_attempt_value : int
        number of current attempt (used for comparing training curves for various hyperparameters)
    best_map_log_file_path : str or None
        file path to best map log file
    """
    def __init__(self,
                 checkpoint_file_name_prefix="model",
                 last_checkpoint_file_name_suffix="last",
                 best_checkpoint_file_name_suffix=None,
                 last_checkpoint_dir_path="",
                 best_checkpoint_dir_path=None,
                 last_checkpoint_file_count=2,
                 best_checkpoint_file_count=2,
                 checkpoint_file_save_callback=None,
                 checkpoint_file_exts=(".params",),
                 save_interval=1,
                 num_epochs=-1,
                 param_names=None,
                 acc_ind=0,
                 # bigger=[True],
                 # mask=None,
                 score_log_file_path=None,
                 score_log_attempt_value=1,
                 best_map_log_file_path=None):

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

        assert (type(param_names) == list)
        self.param_names = param_names

        assert (acc_ind >= 0) and (acc_ind < len(param_names))
        self.acc_ind = acc_ind

        # assert isinstance(bigger, list)
        # self.bigger = np.array(bigger)
        # if mask is None:
        #     self.mask = np.ones_like(self.bigger)
        # else:
        #     assert isinstance(mask, list)
        #     assert (len(mask) == len(bigger))
        #     self.mask = np.array(mask)

        if score_log_file_path is not None:
            self.score_log_file_exist = (os.path.exists(score_log_file_path) and
                                         os.path.getsize(score_log_file_path) > 0)
            self.score_log_file = open(score_log_file_path, "a")
            if not self.score_log_file_exist:
                titles = ["Attempt", "Epoch"] + self.param_names
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
                titles = ["Attempt", "Epoch", self.param_names[self.acc_ind]]
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

    def epoch_test_end_callback(self,
                                epoch1,
                                params,
                                **kwargs):
        curr_acc = params[self.acc_ind]
        if self.can_save:
            last_checkpoint_params_file_stem = None
            if (epoch1 % self.save_interval == 0) or (epoch1 == self.num_epochs):
                last_checkpoint_params_file_stem = self._get_last_checkpoint_params_file_stem(epoch1, curr_acc)
                self.checkpoint_file_save_callback(last_checkpoint_params_file_stem, **kwargs)

                self.last_checkpoint_params_file_stems.append(last_checkpoint_params_file_stem)
                if len(self.last_checkpoint_params_file_stems) > self.last_checkpoint_file_count:
                    removed_checkpoint_file_stem = self.last_checkpoint_params_file_stems[0]
                    for ext in self.checkpoint_file_exts:
                        removed_checkpoint_file_path = removed_checkpoint_file_stem + ext
                        if os.path.exists(removed_checkpoint_file_path):
                            os.remove(removed_checkpoint_file_path)
                    del self.last_checkpoint_params_file_stems[0]

            if (self.best_eval_metric_value is None) or (curr_acc < self.best_eval_metric_value):
                self.best_eval_metric_value = curr_acc
                self.best_eval_metric_epoch = epoch1
                best_checkpoint_params_file_stem = self._get_best_checkpoint_params_file_stem(epoch1, curr_acc)

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
                    self.best_map_log_file.write('\n{:02d}\t{:04d}\t{:.4f}'.format(
                        self.score_log_attempt_value, epoch1, curr_acc))
                    self.best_map_log_file.flush()
        if self.score_log_file is not None:
            score_log_file_row = "\n" + "\t".join([str(self.score_log_attempt_value), str(epoch1)] +
                                                  list(map(lambda x: "{:.4f}".format(x), params)))
            self.score_log_file.write(score_log_file_row)
            self.score_log_file.flush()

    @staticmethod
    def _create_checkpoint_file_path_full_prefix(checkpoint_dir_path,
                                                 checkpoint_file_name_prefix,
                                                 checkpoint_file_name_suffix):
        checkpoint_file_name_full_prefix = checkpoint_file_name_prefix
        if checkpoint_file_name_suffix is not None:
            checkpoint_file_name_full_prefix += ("_" + checkpoint_file_name_suffix)
        return os.path.join(
            checkpoint_dir_path,
            checkpoint_file_name_full_prefix)

    @staticmethod
    def _get_checkpoint_params_file_stem(checkpoint_file_path_prefix, epoch, acc):
        return "{}_{:04d}_{:.4f}".format(checkpoint_file_path_prefix, epoch, acc)

    def _get_last_checkpoint_params_file_stem(self, epoch, acc):
        return self._get_checkpoint_params_file_stem(self.last_checkpoints_prefix, epoch, acc)

    def _get_best_checkpoint_params_file_stem(self, epoch, acc):
        return self._get_checkpoint_params_file_stem(self.best_checkpoints_prefix, epoch, acc)
