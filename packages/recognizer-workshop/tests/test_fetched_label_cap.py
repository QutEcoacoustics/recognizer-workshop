from copy import deepcopy

from recognizer_workshop.app.agile_engine import cap_fetched_label_examples


def _result_set(source_id, label, positive_offsets, negative_offsets):
    return {
        "filter_params": {"project_id": 1147},
        "labels": {
            source_id: {
                label: {
                    "pos": positive_offsets,
                    "neg": negative_offsets,
                }
            }
        },
    }


def _count_examples(result_sets, label, label_type):
    return sum(
        len(type_data.get(label_type, []))
        for result_set in result_sets
        for labels_data in result_set["labels"].values()
        for label_name, type_data in labels_data.items()
        if label_name == label
    )


def test_cap_fetched_label_examples_caps_each_label_type_globally():
    result_sets = [
        _result_set(
            "source-a",
            "Powerful Owl",
            list(range(120)),
            list(range(200, 320)),
        ),
        _result_set(
            "source-b",
            "Powerful Owl",
            list(range(120, 180)),
            list(range(320, 380)),
        ),
    ]
    original = deepcopy(result_sets)

    capped = cap_fetched_label_examples(result_sets, max_examples=100)

    assert _count_examples(capped, "Powerful Owl", "pos") == 100
    assert _count_examples(capped, "Powerful Owl", "neg") == 100
    assert result_sets == original


def test_cap_fetched_label_examples_keeps_smaller_groups_intact():
    result_sets = [
        _result_set("source-a", "Eastern Whipbird", [1, 2], [3]),
        _result_set("source-b", "Eastern Whipbird", [4], [5, 6]),
    ]

    capped = cap_fetched_label_examples(result_sets, max_examples=100)

    assert _count_examples(capped, "Eastern Whipbird", "pos") == 3
    assert _count_examples(capped, "Eastern Whipbird", "neg") == 3
