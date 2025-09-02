from pathlib import Path

from recognizer_workshop.perch_hoplite_custom_modifications import brutalism
from recognizer_workshop.app.services.inference.helpers import make_site_map
from baw_helpers import baw_utils

from recognizer_workshop.perch_hoplite_custom_modifications import classifier as classifier_mod

def run_inference(classifier,
                  db, 
                  output_filepath: str | Path, 
                  baw_api,
                  threshold=0.0, 
                  labels=None, 
                  dataset=None, 
                  subset=1, 
                  exclude_labeled=False):
    """
    Calls the classifier.write_inference_csv() function to write the inference results to a csv file
    injecting the site_id and link columns via a row_func that uses baw_utils and 
    """

    if subset is None:
        subset = 1.0
    subset = float(subset)

    if subset <= 0 or subset > 1:
        raise ValueError('Subset must be between 0 and 1')
    
    embedding_ids = brutalism.get_brute_search_ids(db, subset, exclude_labeled = exclude_labeled)

    # if subset < 1:
    #     n_samples = int(subset * len(all_embedding_ids))
    #     indices = np.arange(len(all_embedding_ids))
    #     np.random.shuffle(indices)
    #     selected_indices = sorted(indices[:n_samples])
    #     embedding_ids = all_embedding_ids[selected_indices]
    # else:
    #     embedding_ids = all_embedding_ids

    print(f'classifying {len(embedding_ids)} ({round(subset*100,0)}%) embeddings from dataset {dataset} ({round(len(embedding_ids) / 12 / 60, 2)} hours) with threshold {threshold}')

    Path(output_filepath).parent.mkdir(parents=True, exist_ok=True)



    site_lookup = make_site_map(db, baw_api)

    def row_func(row=None):
        if row is None:
            return ['end_offset', 'audio_recording_id', 'link', 'site_id', 'site_name']
        baw_domain, arid  = baw_utils.extract_arid_and_domain(row[2])
        link = baw_utils.make_baw_listen_link_from_arid(arid, row[3], 5.0, baw_domain)
        site_info = site_lookup.get(int(arid), '')
        end_offset = row[3] + 5.0  # assuming the window size is 5 seconds
        return [end_offset, arid, link, site_info['site_id'], site_info['site_name']]
        
    classifier_mod.write_inference_csv(classifier,
                                    db,
                                    str(output_filepath),
                                    threshold,
                                    labels,
                                    embedding_ids,
                                    row_func=row_func)
