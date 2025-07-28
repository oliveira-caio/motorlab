# motorlab
this package contains all the tools i've used and all the experiments i've tried so far in my phd project.

# improvement suggestions

## 🟡 architecture & design improvements
1. **model configuration complexity** - model.py still 891 lines but now well-structured with clear separation
2. **hardcoded constants** - magic numbers throughout (seq_length=20, stride=20, etc.)
3. **data loading repetition** - same pattern repeated in train/evaluate functions

## 🟢 code quality enhancements
1. **error handling** - limited error handling for file operations

## ⚡ performance optimizations
2. **memory efficiency** - large data concatenations could be optimized
3. **gpu memory management** - no explicit gpu memory cleanup in training loops
4. **data pipeline** - multiple data transformations could be cached

## 📁 project structure improvements
5. **module organization** - utils.py likely overloaded with diverse utilities
6. **configuration management** - config handling could be centralized

## 🧪 testing & validation
7. **unit tests** - no visible test infrastructure
8. **data validation** - no validation for data shape consistency across sessions

# to do
- add a nondeterministic dataset by selecting the starting points of the sequences randomly.
- figure out a good way to train models with sequences of varying length.
- add early stop such that, if the main metric doesn't improve in 50 iterations, it stops.
- add logging for training so that i can check what was the best validation epoch for a given run.
- check if it's possible to not have one dataset per session, but instead add the session as a third component of the tensors.