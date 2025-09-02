from typing import Iterable, Literal, Optional
from pathlib import Path
import numpy as np

import ipywidgets as widgets
from IPython.display import display
from recognizer_workshop.ui.notebook import agile_widgets
from recognizer_workshop.perch_hoplite_custom_modifications import embedding_display
from matplotlib import pyplot as plt
from perch_hoplite.db import interface as hoplite_interface
from recognizer_workshop.app.agile_modeling_helpers import list_audio_files



class NotebookUI:
    """Stateful notebook UI adapter that wraps perch-hoplite UI widgets and stores EDGs per label."""

    def __init__(self,
                 annotator_id: str,
                 sample_rate_hz: int = 32000,
                 frame_rate: int = 100):
        self.annotator_id = annotator_id
        self.sample_rate_hz = sample_rate_hz
        self.frame_rate = frame_rate
        self._edg_by_label: dict[str, embedding_display.EmbeddingDisplayGroup] = {}
        self._active_label: Optional[str] = None
        self._last_query_samples: Optional[np.ndarray] = None


    # -------- Label selection --------
    def pick_label(self, eng, initial_species: str | None = None,
                   initial_call_types: list[str] | None = None) -> None:
        # Use your existing widget; bind to eng.label
        if initial_call_types is None:
            initial_call_types = []
        agile_widgets.label_selector(
            baw_api=eng.baw_api,
            binding=eng.label,
            initial_species=initial_species,
            initial_call_types=initial_call_types,
        )

    # -------- Query window selection --------
    def select_query_window(self, uri: str, window_size_s: float = 5.0) -> np.ndarray | None:
        # normalize Xeno-Canto id and relative path resolution happens outside here
        if uri.startswith('XC'):
            uri = 'xc' + uri[2:]

        qd = embedding_display.QueryDisplay(
            uri=str(uri),
            offset_s=0.0,
            window_size_s=window_size_s,
            sample_rate_hz=self.sample_rate_hz,
        )
        qd.display_interactive()   # user moves slider
        audio_samples = qd.get_audio_window()
        self._last_query_samples = audio_samples
        return audio_samples


    # -------- Show a few labeled examples --------
    def show_examples(self, examples: list[dict], eng) -> None:
        # examples: [{'eid','dataset_name','source_id','offset','score'}, ...]
        members = []
        for ex in examples:
            members.append(
                embedding_display.EmbeddingDisplay(
                    embedding_id=ex['eid'],
                    dataset_name=ex['dataset_name'],
                    uri=ex['source_id'],
                    offset_s=ex['offset'],
                    score=ex.get('score', 0.0),
                    sample_rate_hz=self.sample_rate_hz,
                    frame_rate=self.frame_rate,
                )
            )
        edg = embedding_display.EmbeddingDisplayGroup.create(
            members=members,
            sample_rate_hz=self.sample_rate_hz,
            audio_loader=eng.audio_filepath_loader,
        )
        edg.baw_config = eng.config.baw_config
        edg.display(positive_labels=[])  # just visualize; no active-label stored


    # -------- Show search results and remember EDG(s) --------
    def show_results(self, eng, results_or_label) -> None:
        # Accept either a TopKSearchResults object, or a label key to pull from eng.search_results
        if hasattr(results_or_label, "search_results"):
            # No label key provided; try to infer from engine target labels
            label_key = (eng.get_target_labels() or ["_"])[0]
            results = results_or_label
        else:
            label_key = str(results_or_label)
            results = eng.search_results.get(label_key)
            if results is None:
                self.notify(f"No search results stored for label '{label_key}'", "warn")
                return

        edg = embedding_display.EmbeddingDisplayGroup.from_search_results(
            results=results,
            db=eng.db,
            sample_rate_hz=self.sample_rate_hz,
            frame_rate=self.frame_rate,
            audio_loader=eng.audio_filepath_loader,
        )
        edg.baw_config = eng.config.baw_config
        # Show buttons with that label selected as "positive option"
        edg.display(positive_labels=[label_key])

        self._edg_by_label[label_key] = edg
        self._active_label = label_key


    # -------- Harvest labels from the active EDG --------
    def harvest_labels(self) -> Iterable[hoplite_interface.Label]:
        if not self._active_label:
            return []
        edg = self._edg_by_label.get(self._active_label)
        if not edg:
            return []
        return edg.harvest_labels(self.annotator_id)


    # -------- Histogram viewer --------
    def plot_scores_for(self, eng, label: str) -> None:
        scores = None
        # engine tweak below will populate eng.search_scores[label]
        if getattr(eng, "search_scores", None) and label in eng.search_scores:
            scores = eng.search_scores[label]
        if scores is None:
            self.notify(f"No stored scores for label '{label}'. Run search first.", "warn")
            return
        plt.figure()
        _ = plt.hist(scores, bins=100)
        # mark current hits if available
        if eng.search_results and label in eng.search_results:
            hit_scores = [r.sort_score for r in eng.search_results[label].search_results]
            plt.scatter(hit_scores, np.zeros_like(hit_scores), marker='|', alpha=0.5)
        plt.title(f"Score distribution — {label}")
        plt.show()


    # -------- Utilities --------
    def set_active_label(self, label: str | None) -> None:
        self._active_label = label

    def clear_results(self) -> None:
        self._edg_by_label.clear()
        self._active_label = None

    def confirm_overwrite(self, db_path: Path) -> bool:
        # Could wire an ipywidget dialog; default accept for now
        return True

    def notify(self, msg: str, level: Literal["info","warn","error"] = "info") -> None:
        prefix = {"info": "ℹ️", "warn": "⚠️", "error": "❌"}.get(level, "ℹ️")
        print(f"{prefix} {msg}")




# def show_label_selector(engine, initial_species=None, initial_call_types: Union[None, str, list]=None):
#     """
#     Displays a label selector widget that allows the user to select a label from the list of available labels.
#     If a label is provided, it will be used to initialize the widget (if it is valid)"""

#     if initial_call_types is None:
#         initial_call_types = []
#     elif type(initial_call_types) is str:
#         initial_call_types = [initial_call_types]
#     agile_widgets.label_selector(
#         baw_api=engine.baw_api, binding=engine.label, initial_species=initial_species, initial_call_types=initial_call_types
#     )

# def display_query(self, query_uri: str | int | Path): 
#     """
#     Displays the query audio so that the user can select the 5s window to embed
#     """

#     query_uri = str(query_uri)

#     #xeno canto ids are uppercase on the website, audio loader requires lowercase
#     if query_uri.startswith('XC'):
#         query_uri = 'xc' + query_uri[2:]

#     # if the query is a path relative to the self.conifg.labeled_examples_folder, resolve its full path
#     if (self.config.labeled_examples_folder / Path(query_uri)).exists():
#         query_uri = Path(self.config.labeled_examples_folder) / Path(query_uri)

#     # if the query is an integer it's an index into the list of audio files in the labeled_examples_folder
#     elif str(query_uri).isdigit():
#         file_list =  list_audio_files(self.config.labeled_examples_folder, quiet=True)
#         if int(query_uri) < len(file_list):
#             query_uri = file_list[int(query_uri)]
#         else:
#             raise ValueError(f'Index {query_uri} is out of range of the labeled examples folder')
        
#     self.query_display = embedding_display.QueryDisplay(
#         uri=str(query_uri), offset_s=0.0, window_size_s=5.0, sample_rate_hz=32000)
    
#     self.query_display.display_interactive()



# def show_examples(examples, audio_filepath_loader, baw_config):
#     """
#     Show a few random spectrograms of labeled examples, largely for sanity-checking. 
#     """
    


#     display_group_members = []
#     for ex in examples:
#         emb_disp = embedding_display.EmbeddingDisplay(ex['eid'], 
#                                                       ex['dataset_name'], 
#                                                       ex['source'], 
#                                                       ex['offset'], 
#                                                       ex['score'])
#         display_group_members.append(emb_disp)
#         edg = embedding_display.EmbeddingDisplayGroup.create(display_group_members, 
#                                                                 sample_rate_hz=32000, 
#                                                                 audio_loader=audio_filepath_loader)
        
#         # TODO: in the old perch repo, i modified the embeddingDisplayGroup to have a baw_config
#         # and use baw to load audio if the baw-config is set
#         # this is not implemented in the new perch-hoplite. Need to find another way around this. 
#         print("Warning: baw_config is not set in the display group - so this is probably going to fail")
#         edg.baw_config = baw_config
#         edg.display(positive_labels=[])



# def display_search_results(engine):
#     """
#     Displays the results in self.search_results (which come from either classifier or single query searching)
#     """

#     #query_label = self.label['value']

#     # if not query_label:
#     #     print('No label set. Please select a label before displaying search results.')
#     #     return
    
#     plt.figure()

#     for query_label, results in engine.search_results.items():
#         display_group = embedding_display.EmbeddingDisplayGroup.from_search_results(
#             results, engine.db, sample_rate_hz=32000, frame_rate=100,
#             audio_loader=engine.audio_filepath_loader)
        
#         # injecting this into the embedding_display_group causes it to load audio from baw
#         display_group.baw_config = engine.config.baw_config
#         display_group.display(positive_labels=[query_label])

#     return display_group

# def get_verification_labels(self):
    
#     if not self.display_group:
#         print('No display group found. Please run display_search_results() first.')
#         return None
    
#     return self.display_group.harvest_labels(self.config.annotator_id)


# def search_results_histogram(engine):

#     _ = plt.hist(engine.all_scores, bins=100)
#     hit_scores = [r.sort_score for r in engine.results.search_results]
#     plt.scatter(hit_scores, np.zeros_like(hit_scores), marker='|',
#                 color='r', alpha=0.5)
#     plt.show()

