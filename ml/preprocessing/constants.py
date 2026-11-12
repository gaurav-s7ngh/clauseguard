"""
ClauseGuard Preprocessing Constants and Category Mappings
=========================================================
Central configuration for Stage 2 dataset preparation:
- 9 selected MVP categories
- Mapping from raw CUAD labels
- Splitting seeds and targets
- Target token and word thresholds for transformer chunking
"""

from typing import Dict, List

# Fixed random seed for complete reproducibility
RANDOM_SEED: int = 42

# Official contract counts
EXPECTED_TOTAL_CONTRACTS: int = 510
EXPECTED_OFFICIAL_TRAIN_CONTRACTS: int = 408
EXPECTED_OFFICIAL_TEST_CONTRACTS: int = 102

# Target internal development split of the 408 train contracts
TARGET_TRAIN_CONTRACTS: int = 340
TARGET_VAL_CONTRACTS: int = 68

# Chunking thresholds (tokenizer-independent word bounds)
# Legal text typically averages ~1.30 - 1.40 subwords per word.
# Target: ~250-350 tokens -> ~180-260 words.
TARGET_MAX_WORDS_PER_CHUNK: int = 240
MAX_WORDS_HARD_LIMIT: int = 360
SENTENCE_OVERLAP_WORDS: int = 35

# 9 Selected ClauseGuard MVP Categories
LABEL_NAMES: List[str] = [
    "liability_cap",
    "termination_for_convenience",
    "anti_assignment",
    "renewal_term",
    "governing_law",
    "audit_rights",
    "ip_ownership_assignment",
    "license_grant",
    "non_compete",
]

NUM_LABELS: int = len(LABEL_NAMES)
LABEL_TO_IDX: Dict[str, int] = {name: idx for idx, name in enumerate(LABEL_NAMES)}
IDX_TO_LABEL: Dict[int, str] = {idx: name for idx, name in enumerate(LABEL_NAMES)}

# Canonical mapping from CUAD raw category name (lowercased) to ClauseGuard label
CUAD_TO_CLAUSEGUARD: Dict[str, str] = {
    "cap on liability": "liability_cap",
    "termination for convenience": "termination_for_convenience",
    "anti-assignment": "anti_assignment",
    "renewal term": "renewal_term",
    "governing law": "governing_law",
    "audit rights": "audit_rights",
    "ip ownership assignment": "ip_ownership_assignment",
    "license grant": "license_grant",
    "non-compete": "non_compete",
}

# Label semantics definitions
LABEL_POSITIVE: int = 1
LABEL_NEGATIVE: int = 0
LABEL_UNKNOWN: int = 0  # When unknown, label value is 0 but supervision mask is 0

MASK_SUPERVISED: int = 1
MASK_UNSUPERVISED: int = 0
