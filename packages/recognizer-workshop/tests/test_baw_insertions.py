from ml_collections import config_dict

from recognizer_workshop.app.services.baw_insertions import (
    get_baw_insertions,
    get_last_filter_for_dataset,
    record_embeddings_added,
)


class _FakeDB:
    def __init__(self):
        self._metadata = {}
        self.commit_calls = 0

    def get_metadata(self, key):
        if key not in self._metadata:
            raise KeyError(key)
        return self._metadata[key]

    def insert_metadata(self, key, value):
        self._metadata[key] = value

    def commit(self):
        self.commit_calls += 1


def test_get_baw_insertions_returns_empty_when_missing():
    db = _FakeDB()
    assert get_baw_insertions(db) == {"datasets": {}}


def test_record_embeddings_added_creates_first_entry():
    db = _FakeDB()
    filt = {"projects.id": {"in": [123]}}

    record_embeddings_added(db, "search_set", filt, 7)

    log_data = get_baw_insertions(db)
    assert log_data["datasets"]["search_set"]["insertions"] == [
        {"filter": filt, "embeddings_added": 7}
    ]
    assert db.commit_calls == 1


def test_record_embeddings_added_increments_semantically_equivalent_filter():
    db = _FakeDB()
    first = {"projects.id": {"in": [123, 234]}}
    same_semantics = {"projects.id": {"in": [234, 123]}}

    record_embeddings_added(db, "search_set", first, 3)
    record_embeddings_added(db, "search_set", same_semantics, 5)

    log_data = get_baw_insertions(db)
    insertions = log_data["datasets"]["search_set"]["insertions"]
    assert len(insertions) == 1
    assert insertions[0]["embeddings_added"] == 8
    assert db.commit_calls == 2


def test_record_embeddings_added_appends_for_distinct_filter():
    db = _FakeDB()
    f1 = {"projects.id": {"in": [123]}}
    f2 = {"projects.id": {"in": [123, 234]}}

    record_embeddings_added(db, "search_set", f1, 2)
    record_embeddings_added(db, "search_set", f2, 4)

    log_data = get_baw_insertions(db)
    insertions = log_data["datasets"]["search_set"]["insertions"]
    assert len(insertions) == 2
    assert insertions[0]["embeddings_added"] == 2
    assert insertions[1]["embeddings_added"] == 4


def test_get_last_filter_for_dataset_uses_fallback_dataset():
    db = _FakeDB()
    metadata = {
        "datasets": {
            "search_set": {
                "insertions": [
                    {"filter": {"projects.id": {"in": [11]}}, "embeddings_added": 10},
                    {"filter": {"projects.id": {"in": [22]}}, "embeddings_added": 12},
                ]
            }
        }
    }
    db.insert_metadata("baw_insertions", config_dict.ConfigDict(metadata))

    got = get_last_filter_for_dataset(db, "fetched_labels", fallback_dataset="search_set")
    assert got == {"projects.id": {"in": [22]}}
