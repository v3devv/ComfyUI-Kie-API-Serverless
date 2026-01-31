def _validate_api_key(api_key: str) -> str:
    """Validate and return the API key. Raises if empty."""
    key = api_key.strip() if api_key else ""
    if not key:
        raise RuntimeError(
            "KIE API key is required. Please enter your API key in the node input."
        )
    return key
