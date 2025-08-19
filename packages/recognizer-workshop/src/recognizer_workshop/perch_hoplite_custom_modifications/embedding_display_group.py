import dataclasses
from typing import Iterator

from perch_hoplite.agile.embedding_display import EmbeddingDisplayGroup as phedg
from perch_hoplite.agile.embedding_display import EmbeddingDisplay
from perch_hoplite import audio_io

from recognizer_workshop import baw_utils

@dataclasses.dataclass
class EmbeddingDisplayGroup(phedg):
  """ extends the EmbeddingDisplayGroup to add custom modifications 
     for displaying embeddings from baw sources """
  
  # if set, this will be used to load audio from instead of the audio loader from file
  baw_config = None
  
  def iterator_with_audio(
      self,
      current_page_only: bool = False,
  ) -> Iterator[EmbeddingDisplay]:
    """Fetch audio for members of the group and compute spectrograms."""
    if len(self.members) > self.results_per_page and current_page_only:
      min_idx = self.current_page * self.results_per_page
      max_idx = min(
          (self.current_page + 1) * self.results_per_page, len(self.members)
      )
      targets = self.members[min_idx:max_idx]
    else:
      targets = self.members

    # We want to iterate over all members, but only load audio for those that
    # don't already have it.
    needs_audio = [m.audio is None for m in targets]
    needs_audio_targets = [m for m in targets if m.audio is None]

    filepaths = [m.uri for m in needs_audio_targets]
    offsets = [m.offset_s for m in needs_audio_targets]
    # audio_iter_ = audio_io.multi_load_audio_window(
    #     filepaths, offsets, self.audio_loader
    # )

    if hasattr(self, 'baw_config') and self.baw_config is not None:
      if 'auth_token' not in self.baw_config:
        self.baw_config['auth_token'] = None
      audio_iter_ = baw_utils.multi_load_baw_audio(
          filepaths=[baw_utils.make_baw_audio_url_from_file_id(fp, offset, 5.0, self.baw_config['domain']) for fp, offset in zip(filepaths, offsets)],
          offsets=offsets,
          auth_token=self.baw_config['auth_token'],
      )
    else:
      audio_iter_ = audio_io.multi_load_audio_window(
          filepaths, offsets, self.audio_loader
      )

    for member, is_dispatched in zip(targets, needs_audio):
      if is_dispatched:
        got_audio = next(audio_iter_)
        if got_audio is not None:
          member.audio = got_audio
          member.spectrogram = self.melspec_layer(got_audio)
      yield member