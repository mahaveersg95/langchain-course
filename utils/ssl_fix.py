import ssl
import os
import httpx
import requests
import requests.adapters

def apply_ssl_fix():
    """Disables SSL verification for corporate proxy networks."""
    ssl._create_default_https_context = ssl._create_unverified_context
    os.environ["PYTHONHTTPSVERIFY"] = "0"

    requests.packages.urllib3.disable_warnings()
    original_send = requests.adapters.HTTPAdapter.send
    def patched_send(self, *args, **kwargs):
        kwargs['verify'] = False
        return original_send(self, *args, **kwargs)
    requests.adapters.HTTPAdapter.send = patched_send

def get_http_client():
    """Returns an httpx client with SSL verification disabled."""
    return httpx.Client(verify=False)
