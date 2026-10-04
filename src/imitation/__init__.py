"""The ``imitation`` family: a model trained from a frozen corpus of decisions (tm-brain), not a rollout.

One fit = one run: read the corpus once, split it once by whole tape from a seed, train on the train tapes only, score the
held-out tapes, early-stop on them, write the same ledger documents every other family writes. A held-out tape that is
also in train fails the fit.
"""
