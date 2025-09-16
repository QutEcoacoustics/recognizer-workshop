from pathlib import Path

from recognizer_workshop.app.agile_modeling_helpers import get_or_create_workshop_provenance 
from baw_helpers.baw_helpers import annotation_upload_csv

def upload_detections(csv_path,
                      upload_name,
                      project_id=None,
                      limit=None,
                      baw_api=None) -> (bool, str | None):
    """
    Uploads the detections from a csv file to the BAW API.
    cvs_path: path to the csv file containing the detections
    project_id: the ID of the project. This is only used for the verify URL.
                if not provided, it will just show a link with a placeholder. 

    """

    if not Path(csv_path).exists():
        print(f'CSV file {csv_path} does not exist. Please provide a valid path.')
        return False, None
    

    provenance_data = get_or_create_workshop_provenance(baw_api)
    provenance_id = provenance_data['id']
    response = annotation_upload_csv(baw_api, csv_path, 
                                        provenance_id=provenance_id,
                                        import_name=upload_name,
                                        commit='true')
    data = response.get('data', {})
    if not data:
        print('Something went wrong with the upload. No data returned from the API.')
        return False, None
    file_upload_id = data.get('id', None)


    if project_id is None:
        project_id = 'placeholder_project_id'
    
    verification_url=f"https://www.ecosounds.org/projects/{project_id}/annotations/verify?importFiles={file_upload_id}"


    print(f"Annotations uploaded successfully. You can view and verify them at the link below:")

    if project_id == 'placeholder_project_id':
        print("You will need to replace the project_id in the URL with a valid one")

    return True, verification_url