"""Browser control primitives."""
import json
import urllib.parse
import urllib.request


def _cdp_get(port, path):
    try:
        resp = urllib.request.urlopen(f"http://localhost:{port}{path}", timeout=5)
        return json.loads(resp.read())
    except Exception as e:
        return {"error": str(e)}


def get_chrome_targets(port=9222):
    """Return all Chrome DevTools Protocol targets via /json. Each target dict
    has {url, title, type, id}. Chrome must be launched with
    --remote-debugging-port=<port>; OSWorld tunnels 9222 by default."""
    return _cdp_get(port, "/json")


def chrome_active_tab(port=9222):
    """Return {url, title, id, tab_count} for the foreground real-page tab,
    filtering chrome://, devtools://, about:, and non-'page' targets.
    Returns {'error': ...} if no normal page is open."""
    targets = get_chrome_targets(port)
    if isinstance(targets, dict):
        return targets
    pages = [
        t for t in targets
        if t.get("type") == "page"
        and not t.get("url", "").startswith(
            ("chrome://", "chrome-extension://", "devtools://", "about:")
        )
    ]
    if not pages:
        return {"error": "no normal pages found"}
    top = pages[0]
    return {
        "url": top.get("url"),
        "title": top.get("title", ""),
        "id": top.get("id"),
        "tab_count": len(pages),
    }


def open_new_tab(url, port=9222):
    """Open `url` in a new Chrome tab via CDP (PUT /json/new?<url>).
    Returns the new target dict (id, url, ...) or {'error': ...}.
    Chrome 122+ requires PUT here — earlier versions accepted GET."""
    quoted = urllib.parse.quote(url, safe=":/?#[]@!$&()*+,;=")
    try:
        req = urllib.request.Request(
            f"http://localhost:{port}/json/new?{quoted}", method="PUT",
        )
        resp = urllib.request.urlopen(req, timeout=5)
        return json.loads(resp.read())
    except Exception as e:
        return {"error": str(e)}
