"Genereric wrapper for the BAW API"
from __future__ import annotations 
from fileinput import filename
import json
import os
import io
import sys
import pytest
import requests
import re
from pathlib import Path
from dataclasses import dataclass, field

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import getpass
import threading
from typing import Any, Optional


from dotenv import load_dotenv, find_dotenv, dotenv_values, set_key

from baw_helpers import utils


# regex to determine default analysis results filenames based on analysis_name patterns
ANALYSIS_FILENAME_MAP = {
    re.compile(r"embeddings", re.I): "embeddings.parquet",   # substring match
}


class baw_api:
    """Generic wrapper for the BAW API"""

    base_url = "https://api.ecosounds.org"
    debug = False
    token = None



    def __init__(self, token=None, base_url=None, debug=False, allow_prompt=None):

       
        if not base_url:
            # If base_url is not provided, use the environment variable or default to ecosounds
            base_url = os.environ.get("API_BASE_URL", "https://api.ecosounds.org")

        debug_level = int(os.environ.get("DEBUG_LEVEL", "0"))
        debug = debug_level > 0

        self.base_url = base_url
        self.debug = debug

        self._tls = threading.local()
        self._lr_lock = threading.Lock()

        # we don't always want to prompt for a token, e.g. non interactive tests
        # if allow_prompt is not specified in either an arg or env var, default to True, 
        # but only if we detect an interactive terminal
        if allow_prompt is None:
            env = os.getenv("BAW_ALLOW_PROMPT")
            if env is not None:
                allow_prompt = env.strip().lower() in {"1","true","yes","on"}
            else:
                allow_prompt = bool(sys.stdin) and sys.stdin.isatty()
                
        self.update_token(token, allow_prompt=allow_prompt)


    def _get_session(self) -> requests.Session:
        """Returns the session for the current thread, creating it if necessary."""
        s = getattr(self._tls, "session", None)
        if s is None:
            s = requests.Session()


            retry = Retry(
                total=3, 
                connect=3, read=3,  
                backoff_factor=0.3,    
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods={"GET", "HEAD"} 
            )
            adapter = HTTPAdapter(pool_connections=20, pool_maxsize=20, max_retries=retry)
            s.mount("https://", adapter)
            s.mount("http://", adapter)
            self._tls.session = s
        return s



    def _auth_headers(self, content_type: str | None = "application/json") -> dict:
        """
        Build headers per request from current token.
        If content_type is None, we omit Content-Type (needed for multipart/form-data).
        """
        h = {}
        if content_type is not None:
            h["Content-Type"] = content_type
        if self.token:
            h["Authorization"] = f'Token token="{self.token}"'
        return h


    def url(self, endpoint):
        """Constructs a full URL for the given endpoint."""
        return f"{self.base_url.rstrip('/')}/{endpoint.lstrip('/')}"


    def get(self, endpoint, params=None, *, timeout: float | tuple | None = None):
        return self.request("GET", endpoint, params=params, timeout=timeout)


    def post(self, endpoint, data=None, params=None, *, timeout: float | tuple | None = None):
        return self.request("POST", endpoint, json=data, params=params, timeout=timeout)


    def baw_filter(self, model: str, 
                   filter_params: dict, 
                   qsp: dict | None = None,
                   ):
        """
        Wrapper for filter endpoints of baw api models. 
        Allows 'slim' filter definition, with no projection or sorting specified, or full filter json. 
        TODO: manual retries. post requests are not retried, however this one should be 
              retried as if it's a get request, since it is idempotent.
        """

        # allow a full filter payload or only the filter portion
        if 'filter' in filter_params:
            filter_payload = filter_params
        else:
            filter_payload = {"filter": filter_params}

        response = self.post(f"/{model}/filter/", data=filter_payload, params=qsp)
        return response

            

    def analysis_result_file(
        self,
        arid: int,
        analysis_job_id: int,
        analysis_name: str,
        filename: str = None,
        *,
        params: dict | None = None,
        save_path: str | None = None,
        timeout: float | tuple | None = None,
        to_memory: bool | None = None,
    ):
        """
        Fetch a an analysis result file.  Same mode rules as download(): exactly one of
        save_path or to_memory=True must be provided.
        """

        if filename is None:

            # if the analysis_name matches the regex in the key of the filename map, set filename to the value of that key
            for key, value in ANALYSIS_FILENAME_MAP.items():
                if re.search(key, analysis_name):
                    filename = value
                    break

            if filename is None:
                raise ValueError("Unable to determine filename from analysis_name")

        ep = f"/analysis_jobs/{analysis_job_id}/results/{arid}/{analysis_name}/{filename}"
        return self.download(ep, save_path, params=params, timeout=timeout, to_memory=to_memory)


    def _send_prepared(
        self,
        method: str,
        url: str,
        *,
        json: dict | list | None = None,
        data: dict | list | bytes | None = None,
        files: dict | None = None,
        params: dict | list[tuple[str, str]] | None = None,
        stream: bool = False,
        timeout: float | tuple | None = None,
        content_type: str | None = None,  # CHANGED: allow override; None for multipart
    ):
        """
        Prepare and send an HTTP request using the thread-local Session.

        Returns:
            (response, prepared_request): On success, a Response and its PreparedRequest.
            On transport errors, returns (None, prepared_request).

        Notes:
            - Does NOT call raise_for_status(); caller decides how to treat 4xx/5xx.
            - If 'files' is provided, we omit Content-Type so requests can set multipart boundary.
            - 'stream' and 'timeout' are passed to send(); mirrors Session.request behavior.
        """
        sess = self._get_session()

        # If 'files' is present, omit Content-Type; requests will set multipart/form-data.
        ct = content_type
        if files is not None:
            ct = None

        headers = self._auth_headers(content_type=ct)
        req = requests.Request(
            method, url, json=json, data=data, files=files, params=params, headers=headers
        )
        prepared_request = sess.prepare_request(req)

        settings = sess.merge_environment_settings(
            prepared_request.url,
            proxies=None,
            stream=stream,
            verify=None,
            cert=None,
        )

        # thread-local "last prepared" for this thread
        self._tls.last_prepared_request = prepared_request

        try:
            response = sess.send(prepared_request, timeout=timeout, **settings)
            return response, prepared_request
        except requests.exceptions.ConnectionError as e:
            print(f"--- CONNECTION ERROR OCCURRED: {e} ---")
            self._print_request_details(prepared_request, response_obj=None, error_context=True)
            return None, prepared_request
        except requests.exceptions.RequestException as e:
            print(f"--- UNEXPECTED REQUESTS ERROR OCCURRED: {e} ---")
            self._print_request_details(prepared_request, response_obj=None, error_context=True)
            return None, prepared_request
        

    def request(        
            self,
            method: str,
            endpoint: str,
            *,
            json: dict | list | None = None,
            data: dict | list | bytes | None = None,
            files: dict | None = None,
            params: dict | list[tuple[str, str]] | None = None,
            stream: bool = False,
            timeout: float | tuple | None = None,
            content_type: str | None = None,
            return_json: bool = True,

        ):
        """
        Wraps the _send_prepared method for easier usage:
        - parses json response if appropriate
        - handles https and json errors and exceptions
        """
        
        url = self.url(endpoint)

        response, prepared_request = self._send_prepared(
            method, url,json=json, data=data, files=files,
            params=params, stream=stream, timeout=timeout,
            content_type=content_type,
        )

        if response is None:
            print("No response received.")
            return None
        
        self._tls.last_response = response
        with self._lr_lock:
            self.last_response = response

        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            print("--- HTTP ERROR OCCURRED ---")
            print(f"Error: {e}")
            # Prefer the failing response.request; fallback to prepared_request if missing
            req_for_log = getattr(getattr(e, "response", None), "request", None) or prepared_request
            self._print_request_details(req_for_log, response_obj=getattr(e, "response", None), error_context=True)
            try:
                if e.response is None:
                    return None
                if return_json:
                    return e.response.json()
                else:
                    return e.response
            except (requests.exceptions.JSONDecodeError, ValueError):
                if self.debug:
                    print("Error response was not JSON.")
                return None
            
        if self.debug:
            print("Request was successful!") 
            self._print_request_details(response.request, response_obj=response, error_context=False)

        if stream or not return_json:
            return response

        try:
            return response.json()
        except (requests.exceptions.JSONDecodeError, ValueError):
            if self.debug:
                print("Warning: Successful request returned non-JSON content (expected JSON).")
            return None


    def post_files(self, endpoint, files=None, data=None, params=None, *, timeout: float | tuple | None = None):
        """
        Upload files using multipart/form-data.
        Reuses request(); we deliberately omit Content-Type so requests sets the boundary.
        """
        return self.request(
            "POST",
            endpoint,
            data=data,
            files=files,
            params=params,
            timeout=timeout,
            content_type=None,   # IMPORTANT: let requests set multipart boundary
            return_json=True,    # keep old behavior: return parsed JSON if possible
        )


    def download(self, endpoint, 
                 save_path: str | None = None, 
                 params=None, 
                 *, 
                 chunk_size: int = 8192, 
                 timeout: float | tuple | None = None, 
                 to_memory: bool | None = None):
        """
        Download a file to disk by streaming the response body.
        Reuses request() with stream=True so we don't load the file into memory.
        Creates directories if they don't exist.
        """

        file_mode = save_path is not None
        mem_mode  = (to_memory is True)
        if file_mode == mem_mode:   # both True or both False
            raise ValueError("Provide 'save_path' where to save file OR set to_memory=True to return a buffer")


        resp = self.request(
            "GET",
            endpoint,
            params=params,
            stream=True,         # stream the body
            timeout=timeout,
            return_json=False,   # we want the raw Response object
        )
        if resp is None:
            return None

        if self.debug:
            print("File download started (streaming).")
            self._print_request_details(resp.request, response_obj=resp, error_context=False)

        try:

            if mem_mode:
                buf = io.BytesIO()
                for chunk in resp.iter_content(chunk_size=chunk_size):
                    if chunk:
                        buf.write(chunk)
                buf.seek(0)
                return buf
            else:
                dirpath = os.path.dirname(save_path)
                if dirpath:                   
                    os.makedirs(dirpath, exist_ok=True)
                with open(save_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=chunk_size):
                        if chunk: 
                            f.write(chunk)
                if self.debug:
                    print(f"File successfully downloaded to: {save_path}")
                return save_path
        except IOError as io_e:
            print(f"--- FILE SAVE ERROR: {io_e} ---")
            # We still have resp.request for details
            self._print_request_details(resp.request, response_obj=resp, error_context=True)
            return None
        finally:
            try:
                resp.close()
            except Exception:
                pass


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
            # content_type = response_obj.headers.get('Content-Type', '').split(';')[0].strip().lower()
            # if content_type.startswith('application/json') or content_type.startswith('text/'):
            #     print("Response Body:")
            #     print(response_obj.text)
            # else:
            #     print(f"Response Body: (Content-Type: {content_type}, likely a file, not printed)")
        else:
            if error_context:
                print("\n--- RESPONSE DETAILS NOT AVAILABLE (e.g., ConnectionError) ---")


    def test_connection(self, silent=True):
        """Test the connection to the BAW API."""
        try:
            # This will use the request method, which has the debug logic
            response = self.request("GET", "/status", return_json=False)
            if not silent:
                print("Connection successful:", response)
            return response
        except requests.RequestException as e:
            if self.debug: # Only print this generic message if debug is on
                print(f"Error connecting to BAW API via test_connection: {e}")
            return None
        

    def test_token(self, default=False):
        """Test the API token on the server for validity."""
        if not self.token:
            if self.debug:
                print("No API token provided. Please set the BAW_AUTH_TOKEN environment variable or pass a token to the baw_api constructor.")
            return False
        
        if self.token[:1] in {"'", '"'} and self.token[-1:] != self.token[:1]:
            # Token in env can be quoted or unquoted, but if it has a missing opening or closing quote, will include the quote in the value
            print("Possibly unclosed quote in your token string")

        r = self.test_connection(silent=True)

        status_code = r.status_code if r is not None else None
        
        if status_code is None:
            is_valid, message = default, "Token can't be verified, no response."
        elif status_code == 401:
            is_valid, message = False, "Token is invalid or expired."
        elif 200 <= status_code < 300:
            is_valid, message = True, "Token is valid."
        elif r is not None and status_code >= 500:
            is_valid, message = default, "Token can't be verified, due to server or connection issues."
        else:
            is_valid, message = default, "Token can't be verified, something went wrong."
        
        if self.debug:
            print(message)
        
        return is_valid
        

    def update_token(self, token=None, save_to_env=True, allow_prompt=True):
        """
        sets the token to the supplied value
        tests the connection. 
        if can't connect, looks for a token in the env. 
        If found, tests it
        if not prompts the user to enter a token
        tests the resulting token
        updates the environment variable if successful
        """

        print("\nUpdating BAW API token...")

        self.token = token

        if not self.test_token():

            print("\nAuthentication token is missing or invalid.")
            env_file = find_dotenv(usecwd=True)
            if not env_file:
                env_file = os.path.join(os.getcwd(), '.env')
                print("No .env file found. A new one will be created if a token is provided.")
                config = {}
            else:
                print(f"Found .env file: {env_file}")
                config = dotenv_values(env_file)

            
            self.token = config.get('BAW_AUTH_TOKEN', None)

            if not self.test_token():

                if not allow_prompt:
                    print("No valid token found and prompting is disabled. Please provide a valid token or set one in the environment.")
                    return

                # Loop until the user provides a valid token or quits
                while allow_prompt:
                    print("\nPlease enter a valid token to proceed. (Type 'quit' to exit)")
                    new_token = getpass.getpass(f"Enter your BAW_AUTH_TOKEN for {self.base_url}: ")
                    
                    if new_token.lower() in ['quit', 'q']:
                        break

                    self.token = new_token 
                    if self.test_token():
                        if save_to_env:
                            set_key(env_file, 'BAW_AUTH_TOKEN', new_token)
                        break 
        else:
            print("\nExisting token from .env file is valid.")


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

