It seems pn.serve(), where a standalone webserver is launched, is not compatible with ipywigets. 

To use ipywigets, we run the panel app from a jupyter notebook, where everything runs through a jupyter kernel. 

Since translating everything to panel widgets is extra work, we will limit ourselves to running through a notebook. 

Components that are in the ipy_widgets folder contain ipywigets. These will only work when run through a notebook. 

With time, we may migrate all these widgets over to panel widgets, allowing them to be run through pn.serve(). 

To allow this transition, a parallel set of wigets in teh pn_widgets folder will exist, and should follow the same api ad the ipy_widgets ones. 