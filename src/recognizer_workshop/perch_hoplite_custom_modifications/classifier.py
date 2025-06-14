import numpy as np
import tqdm
from etils import epath

from perch_hoplite.agile.classifier import LinearClassifier, CsvWorkerState, batched_embedding_iterator, csv_worker_initializer
from perch_hoplite.db import interface as db_interface
from typing import Any, Iterator, Sequence
from concurrent import futures



def csv_worker_fn(
    embedding_ids: np.ndarray, logits: np.ndarray, state: CsvWorkerState
) -> None:
  """Writes a CSV row for each detection.

  Args:
    embedding_ids: The embedding ids to write.
    logits: The logits for each embedding id.
    state: The state of the worker.
  """
  db = state.get_thread_db()
  with epath.Path(state.csv_filepath).open('a') as f:
    for idx, logit in zip(embedding_ids, logits):
      source = db.get_embedding_source(idx)
      for a in np.argwhere(logit > state.threshold):
        lbl = state.labels[a[0]]
        row = [
            idx,
            source.dataset_name,
            source.source_id,
            source.offsets[0],
            lbl,
            logit[a][0],
        ]
        if state['row_func'] is not None:
          row += state['row_func'](row)
        f.write(','.join(map(str, row)) + '\n')


def write_inference_csv(
    linear_classifier: LinearClassifier,
    db: db_interface.HopliteDBInterface,
    output_filepath: str,
    threshold: float,
    labels: Sequence[str] | None = None,
    embedding_ids: np.ndarray | None = None,
    row_func = None
) -> None:
  """Write a CSV for all audio windows with logits above a threshold."""
  if embedding_ids is None:
    embedding_ids = db.get_embedding_ids()
  if labels is None:
    labels = linear_classifier.classes
  else:
    labels = tuple(set(labels).intersection(linear_classifier.classes))
  label_ids = {cl: i for i, cl in enumerate(linear_classifier.classes)}
  target_label_ids = np.array([label_ids[l] for l in labels])
  logits_fn = lambda batch_embs: linear_classifier(batch_embs)[
      :, target_label_ids
  ]
  detection_count = 0
  state = CsvWorkerState(
      db=db,
      csv_filepath=output_filepath,
      labels=labels,
      threshold=threshold,
  )

  # hacking in row_func to allow custom operations on each row
  state.row_func = row_func

  emb_iter = batched_embedding_iterator(db, embedding_ids, batch_size=1024)
  with futures.ThreadPoolExecutor(
      max_workers=1,
      initializer=csv_worker_initializer,
      initargs=(state,),
  ) as executor:
    for batch_idxes, batch_embs in tqdm.tqdm(emb_iter):
      logits = logits_fn(batch_embs)
      # Filter out rows with no detections, avoiding extra database retrievals.
      detections = logits > threshold
      keep_rows = detections.max(axis=1)
      logits = logits[keep_rows]
      kept_idxes = batch_idxes[keep_rows]
      executor.submit(csv_worker_fn, kept_idxes, logits, state)
      detection_count += detections.sum()
  print(f'Wrote {detection_count} detections to {output_filepath}')