from collections import defaultdict

IN_SCOPE_CATEGORIES = ["gad", "depression", "cross_topic"]


def in_scope(gold_set):
    return [item for item in gold_set if item["category"] != "out_of_scope"]


def source_hit(results, item):
    # Did at least one retrieved chunk come from a page that should answer it?
    return any(doc.metadata["source"] in item["expected_source"] for doc in results)


def summarize(rows, metric_names):
    """rows: one dict per question, with "category" and each metric (0-1).
    Returns {category: {metric: mean}}, plus an "overall" entry."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["category"]].append(row)
        groups["overall"].append(row)

    summary = {}
    for category, group in groups.items():
        summary[category] = {"n": len(group)}
        for name in metric_names:
            summary[category][name] = sum(r[name] for r in group) / len(group)
    return summary


def print_summary(summary, metric_names):
    header = f"{'category':<13}{'n':>4}" + "".join(f"{name:>16}" for name in metric_names)
    print(header)
    for category in IN_SCOPE_CATEGORIES + ["overall"]:
        if category not in summary:
            continue
        values = summary[category]
        line = f"{category:<13}{values['n']:>4}"
        line += "".join(f"{values[name]:>16.1%}" for name in metric_names)
        print(line)
