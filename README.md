# motorlab
this package contains all the tools i've used and all the experiments i've tried so far in my phd project.

# to do
- add a nondeterministic dataset by selecting the starting points of the sequences randomly.
- figure out a good way to train models with sequences of varying length.
- add early stop such that, if the main metric doesn't improve in 50 iterations, it stops.
- add logging for training so that i can check what was the best validation epoch for a given run.
- split trials and homing between left and right and balance the datasets.
- move the concatenation from `iterate` to the creation of the datasets.
- check if it's possible to not have one dataset per session, but instead add the session as a third component of the tensors.