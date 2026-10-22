SMALL_MODEL = "openai/gpt-oss-20b"
LARGE_MODEL = "openai/gpt-oss-120b"

# Cumulative risk: score = old * RISK_DECAY + new, crisis at >= RISK_THRESHOLD.
# Sustained rating s levels off at s / (1 - RISK_DECAY), so 0.7 lets repeated
# 2s trigger (ceiling 6.67) while repeated 1s do not (ceiling 3.33).
RISK_DECAY = 0.7
RISK_THRESHOLD = 4
