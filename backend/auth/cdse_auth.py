"""
CDSE Sentinel Hub OAuth2 token management.

Credentials are read exclusively from environment variables:

    CDSE_CLIENT_ID     - OAuth2 client ID from your CDSE dashboard
    CDSE_CLIENT_SECRET - OAuth2 client secret from your CDSE dashboard

Never hard-code credentials. Never commit them to source control.
See acquisition/README.md for account setup instructions.
"""

from __future__ import annotations

import os
import time
import requests


# CDSE OAuth2 token endpoint
CDSE_TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu"
    "/auth/realms/CDSE/protocol/openid-connect/token"
)

# Refresh the token this many seconds before it actually expires
_EXPIRY_MARGIN_SECONDS = 60


class CDSEAuthError(Exception):
    """
    Raised when CDSE authentication fails.

    Possible causes:
        - Missing CDSE_CLIENT_ID / CDSE_CLIENT_SECRET environment variables
        - Invalid or expired OAuth2 client credentials
        - Network error reaching the CDSE identity service
    """


class CDSEToken:
    """
    A single short-lived OAuth2 Bearer token with expiry tracking.

    Tokens issued by CDSE typically expire after 600 seconds (10 minutes).
    CDSEAuth handles automatic renewal.
    """

    def __init__(self, access_token: str, expires_in: int) -> None:
        self.access_token = access_token
        # Use monotonic clock so wall-clock adjustments don't affect validity
        self._expires_at: float = (
            time.monotonic() + expires_in - _EXPIRY_MARGIN_SECONDS
        )

    def is_valid(self) -> bool:
        """True if the token has not expired (with safety margin)."""
        return time.monotonic() < self._expires_at

    def as_bearer_header(self) -> dict[str, str]:
        """Return an Authorization header dict ready for requests."""
        return {"Authorization": f"Bearer {self.access_token}"}


class CDSEAuth:
    """
    Manages OAuth2 client_credentials tokens for CDSE Sentinel Hub APIs.

    Tokens are cached in memory and refreshed automatically when they expire.
    A single CDSEAuth instance can be reused across multiple API calls.

    Usage::

        auth = CDSEAuth()
        headers = {**auth.auth_headers(), "Content-Type": "application/json"}
        response = requests.post(url, headers=headers, json=body)
    """

    def __init__(self) -> None:
        self._client_id: str = os.environ.get("CDSE_CLIENT_ID", "").strip()
        self._client_secret: str = os.environ.get("CDSE_CLIENT_SECRET", "").strip()

        missing: list[str] = []
        if not self._client_id:
            missing.append("CDSE_CLIENT_ID")
        if not self._client_secret:
            missing.append("CDSE_CLIENT_SECRET")

        if missing:
            raise CDSEAuthError(
                "The following required environment variables are not set:\n\n"
                + "\n".join(f"    {v}" for v in missing)
                + "\n\n"
                "See acquisition/README.md — section 'Setting environment variables' "
                "for setup instructions."
            )

        self._token: CDSEToken | None = None

    def get_token(self) -> CDSEToken:
        """
        Return a valid token, fetching a new one if necessary.

        Thread safety: not thread-safe. Use one CDSEAuth per thread for
        concurrent workloads.
        """
        if self._token is None or not self._token.is_valid():
            self._token = self._fetch_new_token()
        return self._token

    def auth_headers(self) -> dict[str, str]:
        """
        Return Authorization headers for an API request.

        Call this immediately before each request so stale tokens are
        renewed automatically.
        """
        return self.get_token().as_bearer_header()

    def _fetch_new_token(self) -> CDSEToken:
        """
        Request a new OAuth2 access token from the CDSE identity service.

        Uses the client_credentials grant type (no user interaction).
        """
        try:
            resp = requests.post(
                CDSE_TOKEN_URL,
                data={
                    "grant_type": "client_credentials",
                    "client_id": self._client_id,
                    "client_secret": self._client_secret,
                },
                timeout=30,
            )
        except requests.RequestException as exc:
            raise CDSEAuthError(
                f"Network error while requesting CDSE token: {exc}"
            ) from exc

        if resp.status_code != 200:
            raise CDSEAuthError(
                f"CDSE token request failed: HTTP {resp.status_code}\n"
                f"Response body: {resp.text[:600]}\n\n"
                "Possible causes:\n"
                "  - CDSE_CLIENT_ID or CDSE_CLIENT_SECRET is wrong\n"
                "  - The OAuth2 client has been revoked in the CDSE dashboard\n"
                "  - The CDSE identity service is temporarily unavailable"
            )

        try:
            payload = resp.json()
        except ValueError as exc:
            raise CDSEAuthError(
                f"Could not parse JSON from token response: {exc}"
            ) from exc

        if "access_token" not in payload:
            raise CDSEAuthError(
                f"Token response does not contain 'access_token'.\n"
                f"Response keys: {sorted(payload.keys())}"
            )

        expires_in = int(payload.get("expires_in", 300))
        print(f"  [AUTH] Token acquired (expires in {expires_in}s).")
        return CDSEToken(payload["access_token"], expires_in)
