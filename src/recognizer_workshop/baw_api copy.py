"Genereric wrapper for the BAW API"

import requests
import json
class baw_api:

    def __init__(self, token, base_url="https://api.ecosounds.org", debug=False):
        self.base_url = base_url
        self.debug = debug
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f'Token token="{token}"',
            "Content-Type": "application/json"
        })

    def url(self, endpoint):
        """Constructs a full URL for the given endpoint."""
        return f"{self.base_url}/{endpoint.lstrip('/')}"

    def get(self, endpoint, params=None):
        """Sends a GET request to the specified endpoint."""
        return self.request("GET", endpoint, params=params)
    
    def post(self, endpoint, data=None):
        """Sends a POST request to the specified endpoint."""
        return self.request("POST", endpoint, data=data)


    def request(self, method, endpoint, data=None, params=None):
        url = self.url(endpoint)
        
        # Prepare the request first. This allows us to inspect it before sending.
        # Use session's prepare_request to include session-level headers etc.
        req = requests.Request(method, url, json=data, params=params)
        prepared_request = self.session.prepare_request(req)

        try:
            # Send the prepared request
            response = self.session.send(prepared_request)
            response.raise_for_status() # This will raise HTTPError for 4xx/5xx responses
            print("Request was successful!") # Print this first so it's clear
            if self.debug:
                self._print_request_details(prepared_request, response_obj=response, error_context=False)

            try:
                json_response = response.json()
                return json_response
            except requests.exceptions.JSONDecodeError:
                print("Warning: Successful request returned non-JSON content.")
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
            print("Response Body (this might contain a server-side traceback!):")
            print(response_obj.text)
        else:
            if error_context: # Only say it's unavailable if it's an error context where response might be missing
                print("\n--- RESPONSE DETAILS NOT AVAILABLE (e.g., ConnectionError) ---")



    def test_connection(self, silent=True):
        """Test the connection to the BAW API."""
        try:
            response = self.get("/status")
            if not silent:
                print("Connection successful:", response)
            return response
        except requests.RequestException as e:
            print(f"Error connecting to BAW API: {e}")
            return None
        
    @staticmethod
    def print_response(response, file=None):
        """Prints the response in a formatted way."""
        formatted_response = json.dumps(response, indent=2)
        if isinstance(response, dict):
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