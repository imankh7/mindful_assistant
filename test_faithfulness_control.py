from faithfulness import judge_faithfulness

source = """GAD may run in families, but the exact cause of the disorder is unknown.
Research suggests that GAD results from a mix of genetics, brain chemistry, biology, and environment.
To be diagnosed with GAD, a person must find it difficult to control worry on most days for at least 6 months."""

cases = [
    (
        "Faithful restatement",
        "The exact cause of GAD is unknown, but it may run in families and involves genetics and environment.",
        True,
    ),
    (
        "Invented statistic",
        "GAD affects approximately 3% of adults in the United States each year.",
        False,
    ),
    (
        "Overstated claim",
        "GAD is hereditary and is passed directly from parents to children.",
        False,
    ),
    (
        "Invented treatment advice",
        "The best approach is to start with CBT for 12 weeks before considering medication.",
        False,
    ),
    (
        "Wrong detail",
        "To be diagnosed with GAD, symptoms must be present for at least 3 months.",
        False,
    ),
]

print("FAITHFULNESS JUDGE — NEGATIVE CONTROL TEST\n")
print("=" * 70)

correct = 0
for label, answer, expected in cases:
    result = judge_faithfulness(answer, source)
    passed = result == expected
    if passed:
        correct += 1
    mark = "PASS" if passed else "FAIL"
    print(
        f"[{mark}] {label}: judge said {'GROUNDED' if result else 'NOT_GROUNDED'} (expected {'GROUNDED' if expected else 'NOT_GROUNDED'})"
    )

print("=" * 70)
print(f"\nJudge accuracy on control set: {correct}/{len(cases)}")
