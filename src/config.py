
SUMMARY_TARGETS = {
    "aclsum": {"sentences": 1, "words": 25},
    "facetsum": {"sentences": 2, "words": 50},
    "pmc": {"sentences": 4, "words": 75},
}

from paths import DATA_ROOT, LOG_ROOT, RESULTS_ROOT

BASE_STORAGE = DATA_ROOT
RESULTS_STORAGE = RESULTS_ROOT
LOG_STORAGE = LOG_ROOT

TYPE_BY_DATASET = {
    "aclsum": "full",
    "facetsum": "sampled",
    "pmc": "sampled",
}