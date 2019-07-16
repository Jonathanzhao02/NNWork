# NNWork
Code used for an IB extended essay to compare mini-batch SGD and LEEA in R. SGD functions much better as it uses Keras functions, and LEEA is roughly coded by hand.

As mentioned, SGD uses Keras functions on GPU so it runs extremely quick (~30 seconds per run on a low-end GPU)
LEEA uses small amounts of parallelization on CPU so it runs extremely non-quick (~20 hours per run)

Note the "runs" match in number of training example evaluations, for example 1000000 evaluations in SGD translates to about 500 generations of 1000 utilizing 2 training samples in LEEA.
