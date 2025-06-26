# motorlab
this package contains all the tools i've used and all the experiments i've tried so far in my phd project.

# to do
- add a nondeterministic dataset by selecting the starting points of the sequences randomly.
- figure out a good way to train models with sequences of varying length.
- copy all the meta from `reponses` to `spike_count` in the `create_spike_count.ipynb` and indices to int.
- when i load a model from a checkpoint, the first epoch should reset to the last epoch otherwise it'll overwrite what i have already saved.
- add early stop such that, if the main metric doesn't improve in 50 iterations, it stops.
- when i train a model with the weights from a frozen module, it should create a new uid and save separately. moreover, there should be one config and checkpoint directories to load the model/config from and one config and checkpoint directories to save the model into.
- add logging for training so that i can check what was the best validation epoch for a given run.