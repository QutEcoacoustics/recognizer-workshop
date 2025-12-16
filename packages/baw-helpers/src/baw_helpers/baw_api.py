"Genereric wrapper for the BAW API"
from __future__ import annotations 
import getpass
import io
import json
import logging
import os
import re
import sys
import threading

import requests
from dotenv import find_dotenv, dotenv_values, set_key
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# regex to determine default analysis results filenames based on analysis_name patterns
ANALYSIS_FILENAME_MAP = {
    re.compile(r"embeddings", re.I): "embeddings.parquet",
}

log = logging.getLogger(__name__)

class AuthenticationError(Exception):
    """Raised when authentication is required but not provided or invalid."""
    pass


class baw_api:
    """Generic client for the BAW API"""

    # defaults
    base_url = "https://api.ecosounds.org"
    token = None


    def __init__(self, token=None, base_url=None, allow_prompt=None):
        """
        Initializes the BAW API client.
        Args:
            token (str): API authentication token. If None, will look for BAW_AUTH_TOKEN in environment variables.
            base_url (str): Base URL for the BAW API. If None, will use the BAW_API_BASE_URL environment variable or default to ecosounds.
            allow_prompt (bool): If True, allows prompting the user for a token if none is provided or found. 
                                 If None, defaults to True if running in an interactive terminal.
        
        """
        start_of_token = token[:4] + "..." if token else "None"
        log.debug(f"BAW API initializing with token={start_of_token}, base_url={base_url}")

        if not base_url:
            # If base_url is not provided, use the environment variable or default to ecosounds
            base_url = os.environ.get("API_BASE_URL", "https://api.ecosounds.org")

        self.base_url = base_url

        self._tls = threading.local()
        self._lr_lock = threading.Lock()

        # we don't always want to prompt for a token, e.g. non interactive tests or panel app
        # if allow_prompt is not specified in either an arg or env var,
        # default to True but only if we detect an interactive terminal
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
        Only adds Authorization header if token is set, otherwise we make unauthenticated requests in anonymous mode.
        """
        log.debug(f"Creating auth headers, token available: {bool(self.token)}")

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

        # raise if in anonumous mode, because results files always need auth
        if self.token is None:
            raise AuthenticationError("Authentication token is required to download analysis result files.")

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
        content_type: str | None = None,
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
            log.error(f"CONNECTION ERROR OCCURRED: {e}")
            self._print_request_details(prepared_request, response_obj=None, error_context=True)
            return None, prepared_request
        except requests.exceptions.RequestException as e:
            log.error(f"UNEXPECTED REQUESTS ERROR OCCURRED: {e}")
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
            log.warning("No response received (Connection Error).")
            raise requests.exceptions.ConnectionError(f"Failed to connect to {url}")
        
        self._tls.last_response = response
        with self._lr_lock:
            self.last_response = response

        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            log.error("HTTP ERROR OCCURRED")
            log.error(f"Error: {e}")
            # Prefer the failing response.request; fallback to prepared_request if missing
            req_for_log = getattr(getattr(e, "response", None), "request", None) or prepared_request
            self._print_request_details(req_for_log, response_obj=getattr(e, "response", None), error_context=True)

            if response.status_code == 401:
                if self.token is None:
                    message = "Authentication token is missing."
                else:
                    message = "Authentication token is invalid or expired."

                raise AuthenticationError(f"401 Unauthorized: {message} {response.text}") from e

            raise e

            
        log.debug("Request was successful!") 
        self._print_request_details(response.request, response_obj=response, error_context=False)

        if stream or not return_json:
            return response

        try:
            return response.json()
        except (requests.exceptions.JSONDecodeError, ValueError):
            log.error("Successful request returned non-JSON content.")
            raise ValueError(f"Response from {endpoint} was not valid JSON")
            


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

        log.info("File download started (streaming).")
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
                log.info(f"File successfully downloaded to: {save_path}")
                return save_path
        except IOError as io_e:
            log.error(f"FILE SAVE ERROR: {io_e}")
            # We still have resp.request for details
            self._print_request_details(resp.request, response_obj=resp, error_context=True)
            raise
        finally:
            try:
                resp.close()
            except Exception:
                pass


    def _print_request_details(self, request_obj, response_obj=None, error_context=False):
        """
        Helper method to log details of the HTTP request and (optionally) response.
        """
        log.debug("\n--- REQUEST/RESPONSE DEBUG INFO ---")
        if error_context:
            log.error("--- CONTEXT: An HTTP ERROR occurred ---")
        else:
            log.debug("--- CONTEXT: successful request ---")

        log.info("\n--- DETAILS OF THE REQUEST SENT ---")
        log.info(f"Method: {request_obj.method}")
        log.info(f"URL:    {request_obj.url}")

        log.debug("\nHeaders Sent:")
        for key, value in request_obj.headers.items():
            # Always hide sensitive Authorization header
            if key.lower() == "authorization":
                log.debug(f"  {key}: [HIDDEN]")
            else:
                log.debug(f"  {key}: {value}")

        log.debug("\nBody/Payload Sent:")
        if request_obj.body:
            try:
                # Try to format it as pretty JSON
                parsed_body = json.loads(request_obj.body.decode('utf-8'))
                log.debug(json.dumps(parsed_body, indent=2))
            except json.JSONDecodeError:
                # If it's not JSON, just print the raw text
                log.debug(request_obj.body.decode('utf-8'))
        else:
            log.debug("No body was sent.")

        if response_obj is not None:
            log.info("\n--- DETAILS OF THE SERVER'S RESPONSE ---")
            log.info(f"Status Code: {response_obj.status_code}")
            log.debug("Response Headers:")
            for key, value in response_obj.headers.items():
                log.debug(f"  {key}: {value}")
        else:
            if error_context:
                log.info("\n--- RESPONSE DETAILS NOT AVAILABLE (e.g., ConnectionError) ---")



        

    def test_token(self, default=False):
        """Test the API token on the server for validity."""
        if not self.token:
            log.warning("No API token provided. Please set the BAW_AUTH_TOKEN environment variable or pass a token to the baw_api constructor.")
            return False
        
        if self.token[:1] in {"'", '"'} and self.token[-1:] != self.token[:1]:
            # Token in env can be quoted or unquoted, but if it has a missing opening or closing quote, will include the quote in the value
            log.warning("Possibly unclosed quote in your token string")

        # return True if token is valid, False if not, or default if we can't verify
        try:
            # /status does not require auth, but will return 401 if token is provided and is invalid
            response = self.request("GET", "/status", return_json=False)
            log.info(f"Connection successful: {response}")
            return True
            
        except requests.RequestException as e:
            # Covers 5xx, 404, DNS failures, Timeouts
            log.error(f"Error connecting to BAW API during test_token: {e}")
            assumption = "valid" if default else "invalid"
            log.warning(f"Token can't be verified, assuming {assumption}.")
            return default
            
        except AuthenticationError:
            log.error("TOKEN INVALID. Connection successful, but Authentication failed.")
            return False
            
        except Exception as e:
            log.error(f"Unexpected error during test_token: {e}")
            assumption = "valid" if default else "invalid"
            log.warning(f"Token can't be verified, assuming {assumption}.")
            return default

    @property
    def hidden_token(self):
        """Returns a masked version of the token for logging."""
        if self.token is None:
            return "None"
        elif len(self.token) <= 8:
            return self.token[0] + "..." + self.token[-1]
        else:
            return self.token[:4] + "..." + self.token[-4]

    def update_token(self, 
                     token=None, 
                     allow_prompt=True, 
                     require_auth=False):
        """
        Validate and set authentication token.
        
        Args:
            token: Explicit token to use
            allow_prompt: Whether to interactively prompt for token
            require_auth: If True, raises error when no valid token available

            If a token is available either passed or from environment, and is invalid, will raise an error whether or not 
        """
        log.info("\nUpdating BAW API token...")
        
        supplied_token = token if token else None

        # token not supplied, see if it's in the environment already
        if supplied_token is None:
            supplied_token = os.getenv('BAW_AUTH_TOKEN')

        # no token in the environment (and not supplied)
        # try loading from dotenv
        env_file = find_dotenv(usecwd=True)
        if not env_file:
            # if there is no env file, we still create a default path
            # because in some situations we will save a inputted token there
            env_file = os.path.join(os.getcwd(), '.env')

        if supplied_token is None:
            if os.path.exists(env_file):
                config = dotenv_values(env_file)
                supplied_token = config.get('BAW_AUTH_TOKEN', None)
                supplied_token = supplied_token if supplied_token else None

        # any empty strings should be converted to None
        if not supplied_token:
            supplied_token = None

        # we have a token, so test it. 
        if supplied_token is not None:
            self.token = supplied_token
            if self.test_token():
                return True 
            else:
                raise AuthenticationError(f"Supplied token {self.hidden_token} is invalid.")
        else:
            log.info("No token supplied in arguments or environment.")
        
        # if we reach here, we have no valid token
        # if we don't require it, just succeed in anonymous mode with a message
        # if we do require it, either prompt or fail
        if require_auth:
            if not allow_prompt:
                raise AuthenticationError("Authentication required but no valid token available")
            else:
                self._prompt_for_token(env_file=env_file)
                if self.token is None:
                    raise AuthenticationError("No valid token provided, exiting.")
            
        if self.token is None:
            log.warning("\n💡 No Valid Token supplied, creating anonymous client")
            log.info("\n💾 For authenticated access initialize the client with a token or:")
            log.info("   Option A: Add to your .env file:")
            log.info("             BAW_AUTH_TOKEN=your-token-here")
            log.info("   Option B: Export as environment variable:")
            log.info("             export BAW_AUTH_TOKEN=your-token-here")

                
        # Will have already returned false if require_auth and no valid token
        # or if a token was supplied but was invalid. 
        return True


    def _prompt_for_token(self, env_file = None):
        """
        Prompts the user to enter a token interactively.
        Sets self.token if it's valid. 
        
        """
        new_token = None
        while True:
            print("\nPlease enter a valid token to proceed. (Type 'quit' to to proceed in anonymous mode)")
            new_token = getpass.getpass(f"Enter your BAW_AUTH_TOKEN for {self.base_url}: ")
            
            if new_token.lower() in ['quit', 'q']:
                return

            self.token = new_token 
            if self.test_token():
                if env_file is not None:
                    # promt to save to env
                    save = input(f"Do you want to save the token to {env_file} for future use? (y/n): ").strip().lower()
                    if save in ['y', 'yes']:
                        set_key(env_file, 'BAW_AUTH_TOKEN', new_token)
                return
            else:
                print("The token you entered is invalid. Please try again.")
        
    

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

        log.info(formatted_response)

