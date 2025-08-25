# motorlab
this package contains all the tools i've used and all the experiments i've tried so far in my phd project.

# to do
- refactor entire codebase
    - inside utils: wandb, current utils.py, but split
- remove multi-gpu support
- create plots while training
- run the test in the end with the final plot
- create dashboards?
- load items from memory instead of saving the entire dataset into cpu if i need it. add caching for this functionality, though.
- refactor model functions to use configuration dataclasses instead of long parameter lists.
- move to experanto.