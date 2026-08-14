import csv
import logging
import numpy as np
import tqdm
import tensorflow as tf
from etils import epath

from perch_hoplite.agile.classifier import (
  LinearClassifier,
  CsvWorkerState,
  batched_embedding_iterator,
  bce_loss,
  hinge_loss,
  eval_classifier,
)
from perch_hoplite.db import interface as db_interface
from typing import Any, Iterator, Sequence
from concurrent import futures

import logging
log = logging.getLogger(__name__)


def train_linear_classifier(
    data_manager,
    learning_rate: float,
    weak_neg_weight: float,
    num_train_steps: int,
    loss: str = "bce",
    l1_strength: float = 0.0,
    l2_strength: float = 0.0,
    regularize_bias: bool = False,
) -> tuple[LinearClassifier, dict[str, Any]]:
  """Train perch-hoplite's linear classifier with optional weight penalties."""
  if l1_strength < 0 or l2_strength < 0:
    raise ValueError("Regularization strengths must be non-negative.")

  embedding_dim = data_manager.db.embedding_dimension()
  num_classes = len(data_manager.get_target_labels())
  lin_model = tf.keras.Sequential([
      tf.keras.Input(shape=[embedding_dim]),
      tf.keras.layers.Dense(num_classes),
  ])
  optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
  lin_model.compile(optimizer=optimizer, loss="binary_crossentropy")

  if loss == "hinge":
    loss_fn = hinge_loss
  elif loss == "bce":
    loss_fn = bce_loss
  else:
    raise ValueError(f"Unknown loss: {loss}")

  @tf.function
  def train_step(y_true, embeddings, is_labeled_mask):
    with tf.GradientTape() as tape:
      logits = lin_model(embeddings, training=True)
      total_loss = loss_fn(y_true, logits, is_labeled_mask, weak_neg_weight)
      kernel = lin_model.trainable_variables[0]
      penalty = (
          l1_strength * tf.reduce_sum(tf.abs(kernel))
          + l2_strength * tf.reduce_sum(tf.square(kernel))
      )
      if regularize_bias:
        bias = lin_model.trainable_variables[1]
        penalty += (
            l1_strength * tf.reduce_sum(tf.abs(bias))
            + l2_strength * tf.reduce_sum(tf.square(bias))
        )
      total_loss = tf.reduce_mean(total_loss) + penalty
    gradients = tape.gradient(total_loss, lin_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, lin_model.trainable_variables))
    return total_loss

  train_ids, eval_ids = data_manager.get_train_test_split()
  train_iter = data_manager.batched_example_iterator(
      train_ids, add_weak_negatives=True, repeat=True
  )
  progress = tqdm.tqdm(enumerate(train_iter), total=num_train_steps)
  update_steps = set(b * (num_train_steps // 100) for b in range(100))

  for step, batch in enumerate(train_iter):
    if step >= num_train_steps:
      break
    step_loss = train_step(batch.multihot, batch.embedding, batch.is_labeled_mask)
    if step in update_steps and num_train_steps >= 100:
      progress.update(n=num_train_steps // 100)
      progress.set_description(f"Loss {step_loss:.8f}")
  progress.clear()
  progress.close()

  params = {
      "beta": lin_model.get_weights()[0],
      "beta_bias": lin_model.get_weights()[1],
  }
  if len(eval_ids) > 0:
    eval_scores = eval_classifier(params, data_manager, eval_ids)
    validation_losses = []
    for batch in data_manager.batched_example_iterator(
        eval_ids, add_weak_negatives=False, repeat=False
    ):
      logits = lin_model(batch.embedding, training=False)
      batch_loss = loss_fn(
          batch.multihot, logits, batch.is_labeled_mask, weak_neg_weight
      )
      validation_losses.append(float(batch_loss.numpy()))
    eval_scores["validation_loss"] = float(np.mean(validation_losses))
  else:
    eval_scores = {
        "eval_preds": np.empty((0, num_classes), dtype=np.float32),
        "eval_labels": np.empty((0, num_classes), dtype=np.float32),
        "validation_loss": float("nan"),
    }
  model_config = data_manager.db.get_metadata("model_config")
  linear_classifier = LinearClassifier(
      beta=params["beta"],
      beta_bias=params["beta_bias"],
      classes=data_manager.get_target_labels(),
      embedding_model_config=model_config,
  )
  return linear_classifier, eval_scores


def csv_worker_initializer_factory(row_func: Any = None) -> None:
  """
  Initialize the CSV worker.
  row_func: optional function. will return extra column headings if called with no arguments.
  """
  csv_header = ['idx', 'dataset_name', 'source_id', 'offset', 'label', 'score']
  extras = row_func() if row_func is not None else []
  csv_header.extend(extras)

  def csv_worker_initializer(state: CsvWorkerState) -> None:
    state.get_thread_db()
    with open(state.csv_filepath, 'w', newline='') as f:
      csv.writer(f, lineterminator='\n').writerow(csv_header)
  
  return csv_worker_initializer


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
  with open(state.csv_filepath, 'a', newline='') as f:
    writer = csv.writer(f, lineterminator='\n')
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
        if state.row_func is not None:
          row += state.row_func(row)
        writer.writerow(row)


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
      initializer=csv_worker_initializer_factory(row_func),
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
  log.info(f'Wrote {detection_count} detections to {output_filepath}')