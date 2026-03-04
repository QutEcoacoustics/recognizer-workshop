from types import SimpleNamespace
from unittest.mock import MagicMock

from perch_hoplite.db.search_results import SearchResult, TopKSearchResults

from recognizer_workshop.app.agile_engine import agile_state


def _make_results(*embedding_ids: int) -> TopKSearchResults:
    results = TopKSearchResults(top_k=len(embedding_ids))
    for i, eid in enumerate(embedding_ids, start=1):
        results.update(SearchResult(embedding_id=eid, sort_score=float(i)))
    return results


class _FakeDB:
    def __init__(self, source_ids_by_eid: dict[int, str]):
        self._source_ids_by_eid = source_ids_by_eid

    def get_embedding_source(self, embedding_id: int):
        source_id = self._source_ids_by_eid[embedding_id]
        return SimpleNamespace(source_id=source_id, dataset_name="search_set", offsets=[0.0])


def test_attach_metadata_batches_by_domain():
    domain = "https://api.ecosounds.org"
    source_ids = {
        1: f"{domain}/audio_recordings/111/original",
        2: f"{domain}/audio_recordings/222/original",
    }
    db = _FakeDB(source_ids)

    mock_client = MagicMock()
    mock_client.post.return_value = {
        "data": [
            {"id": 111, "site_id": 10},
            {"id": 222, "site_id": 20},
        ]
    }

    engine = agile_state(config=MagicMock())
    engine.db = db
    engine.baw_api_clients = {domain: mock_client}

    results = _make_results(1, 2)
    engine.attach_metadata_to_search_results(results)

    mock_client.post.assert_called_once()
    call_args = mock_client.post.call_args.kwargs
    assert call_args["data"]["filter"]["id"]["in"] == [111, 222]

    result_list = list(results)
    assert result_list[0].metadata["domain"] == domain
    assert result_list[0].metadata["arid"] in {111, 222}
    assert result_list[0].metadata["recording_info"]["id"] in {111, 222}

    assert result_list[1].metadata["domain"] == domain
    assert result_list[1].metadata["arid"] in {111, 222}
    assert result_list[1].metadata["recording_info"]["id"] in {111, 222}


def test_attach_metadata_skips_unparseable_source_ids():
    source_ids = {1: "local/audio/file.wav"}
    db = _FakeDB(source_ids)

    mock_client = MagicMock()

    engine = agile_state(config=MagicMock())
    engine.db = db
    engine.baw_api_clients = {"https://api.ecosounds.org": mock_client}

    results = _make_results(1)
    engine.attach_metadata_to_search_results(results)

    mock_client.post.assert_not_called()

    result = list(results)[0]
    assert result.metadata["domain"] is None
    assert result.metadata["arid"] is None
    assert result.metadata["recording_info"] is None
