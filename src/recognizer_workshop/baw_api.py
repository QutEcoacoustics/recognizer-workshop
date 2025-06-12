"Genereric wrapper for the BAW API"

import json
import os
import requests

class baw_api:
    """Generic wrapper for the BAW API"""

    def __init__(self, token, base_url="https://api.ecosounds.org", debug=False):
        self.base_url = base_url
        self.debug = debug
        self.session = requests.Session() # Initialize requests.Session()
        self.session.headers.update({
            "Authorization": f'Token token="{token}"',
            "Content-Type": "application/json"
        })

    def url(self, endpoint):
        """Constructs a full URL for the given endpoint."""
        # Ensure proper URL joining, handling leading/trailing slashes
        return f"{self.base_url.rstrip('/')}/{endpoint.lstrip('/')}"

    def get(self, endpoint, params=None):
        """Sends a GET request to the specified endpoint (expects JSON)."""
        return self.request("GET", endpoint, params=params)

    def post(self, endpoint, data=None):
        """Sends a POST request to the specified endpoint (expects JSON)."""
        return self.request("POST", endpoint, data=data)

    def download(self, endpoint, save_path, params=None):
        """
        Downloads a file from the specified endpoint and saves it to the given path.
        Returns the save_path on success, None on failure.
        """
        url = self.url(endpoint)

        req = requests.Request("GET", url, params=params)
        prepared_request = self.session.prepare_request(req)

        response = None # Initialize response to None
        try:
            # Use stream=True to download large files without loading them entirely into memory
            response = self.session.send(prepared_request, stream=True)
            response.raise_for_status() # Check for HTTP errors (4xx, 5xx)

            if self.debug:
                print("File download successful!")
                self._print_request_details(prepared_request, response_obj=response, error_context=False)

            # Ensure the directory for saving exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            # Save the file in chunks
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Explicitly close the stream response
            response.close()

            print(f"File successfully downloaded to: {save_path}")
            return save_path

        except requests.exceptions.HTTPError as e:
            print("--- HTTP ERROR DURING FILE DOWNLOAD ---")
            print(f"Error: {e}")
            self._print_request_details(e.response.request, response_obj=e.response, error_context=True)
            return None
        except requests.exceptions.ConnectionError as e:
            print(f"--- CONNECTION ERROR DURING FILE DOWNLOAD: {e} ---")
            self._print_request_details(prepared_request, response_obj=None, error_context=True)
            return None
        except requests.exceptions.RequestException as e:
            print(f"--- UNEXPECTED REQUESTS ERROR DURING FILE DOWNLOAD: {e} ---")
            self._print_request_details(prepared_request, response_obj=None, error_context=True)
            return None
        except IOError as io_e:
            print(f"--- FILE SAVE ERROR: {io_e} ---")
            self._print_request_details(prepared_request, response_obj=response, error_context=True)
            return None
        finally:
            # Ensure the response stream is closed even on success or failure
            if response:
                response.close()

    def request(self, method, endpoint, data=None, params=None):
        url = self.url(endpoint)
        
        # Prepare the request: This ensures session-level headers (like Auth) are included
        req = requests.Request(method, url, json=data, params=params)
        prepared_request = self.session.prepare_request(req)

        response = None # Initialize response to None
        try:
            # Send the prepared request
            response = self.session.send(prepared_request)
            response.raise_for_status() # Raises HTTPError for 4xx/5xx responses

            if self.debug:
                print("Request was successful!") # Print this first so it's clear
                self._print_request_details(prepared_request, response_obj=response, error_context=False)

            try:
                json_response = response.json()
                return json_response
            except requests.exceptions.JSONDecodeError:
                # If the status was 2xx but the response body wasn't valid JSON (e.g., a 204 No Content)
                if self.debug: # Only print warning if in debug mode
                    print("Warning: Successful request returned non-JSON content (expected JSON).")
                return None 

        except requests.exceptions.HTTPError as e:
            print("--- HTTP ERROR OCCURRED ---")
            print(f"Error: {e}") 
            self._print_request_details(e.response.request, response_obj=e.response, error_context=True)
            try:
                # Attempt to return JSON from error response if available and valid
                return e.response.json()
            except requests.exceptions.JSONDecodeError:
                # If the error response itself is not JSON
                if self.debug: # Only print this warning if in debug mode
                    print("Error response was not JSON.")
                return None 

        except requests.exceptions.ConnectionError as e:
            print(f"--- CONNECTION ERROR OCCURRED: {e} ---")
            self._print_request_details(prepared_request, response_obj=None, error_context=True)
            return None 

        except requests.exceptions.RequestException as e:
            print(f"--- UNEXPECTED REQUESTS ERROR OCCURRED: {e} ---")
            self._print_request_details(prepared_request, response_obj=None, error_context=True)
            return None 

    def _print_request_details(self, request_obj, response_obj=None, error_context=False):
        """
        Helper method to print details of the HTTP request and (optionally) response.
        """
        print("\n--- REQUEST/RESPONSE DEBUG INFO ---")
        if error_context:
            print("--- CONTEXT: An HTTP ERROR occurred ---")
        else:
            print("--- CONTEXT: Debug mode is active (successful request) ---")

        print("\n--- DETAILS OF THE REQUEST SENT ---")
        print(f"Method: {request_obj.method}")
        print(f"URL:    {request_obj.url}")

        print("\nHeaders Sent:")
        for key, value in request_obj.headers.items():
            # Hide sensitive Authorization header content if not in debug mode
            if key.lower() == "authorization" and not self.debug:
                print(f"  {key}: [HIDDEN]")
            else:
                print(f"  {key}: {value}")

        print("\nBody/Payload Sent:")
        if request_obj.body:
            try:
                # Try to format it as pretty JSON
                parsed_body = json.loads(request_obj.body.decode('utf-8'))
                print(json.dumps(parsed_body, indent=2))
            except json.JSONDecodeError:
                # If it's not JSON, just print the raw text
                print(request_obj.body.decode('utf-8'))
        else:
            print("No body was sent.")

        if response_obj is not None:
            print("\n--- DETAILS OF THE SERVER'S RESPONSE ---")
            print(f"Status Code: {response_obj.status_code}")
            print("Response Headers:")
            for key, value in response_obj.headers.items():
                print(f"  {key}: {value}")

            # Only print response body if it's likely text/JSON, not for large binary files
            content_type = response_obj.headers.get('Content-Type', '').split(';')[0].strip().lower()
            if content_type.startswith('application/json') or content_type.startswith('text/'):
                print("Response Body:")
                print(response_obj.text)
            else:
                print(f"Response Body: (Content-Type: {content_type}, likely a file, not printed)")
        else:
            if error_context:
                print("\n--- RESPONSE DETAILS NOT AVAILABLE (e.g., ConnectionError) ---")


    def test_connection(self, silent=True):
        """Test the connection to the BAW API."""
        try:
            # This will use the request method, which has the debug logic
            response = self.get("/status")
            if not silent:
                print("Connection successful:", response)
            return response
        except requests.RequestException as e:
            # The .request method already handles detailed error printing,
            # so this block just needs to catch and return None
            # or you can add a simple message if you prefer.
            if self.debug: # Only print this generic message if debug is on
                print(f"Error connecting to BAW API via test_connection: {e}")
            return None
        
    @staticmethod
    def print_response(response, file=None):
        """Prints the response in a formatted way."""
        if response is None:
            formatted_response = "None (No response or invalid JSON)"
        elif isinstance(response, (dict, list)): 
            formatted_response = json.dumps(response, indent=2)
        else: 
            formatted_response = str(response)

        if file:
            with open(file, 'w') as f:
                f.write(formatted_response)

        print(formatted_response)


class baw_utils:
    def __init__(self, token):
        self.api = baw_api(token)

    

class VerificationManager:
    def __init__(self, token):
        self.baw = baw_utils(token)
        self.verifications = []

    def get_verifications(self, source_id):
        """Fetch verifications for a given source ID."""
        endpoint = f"/sources/{source_id}/verifications"
        response = self.baw.api.get(endpoint)
        self.verifications = response.get('verifications', [])
        return self.verifications

    def add_verification(self, source_id, verification_data):
        """Add a new verification for a given source ID."""
        endpoint = f"/sources/{source_id}/verifications"
        response = self.baw.api.post(endpoint, data=verification_data)
        return response