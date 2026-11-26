# Claude Code's PreToolUse hook for "This PC" agents: asks Agent S whether this tool use may run.
# Exit 0 runs it; exit 2 blocks it and tells the agent why. Agent S listens on the PC's loopback only,
# and each turn gets its own port and token.
$ErrorActionPreference = 'Stop'
[Console]::InputEncoding = [Text.Encoding]::UTF8
$body = [Console]::In.ReadToEnd()
$answer = ''
try {
    $request = [System.Net.HttpWebRequest]::Create("http://127.0.0.1:$($env:AGENT_S_APPROVAL_PORT)/approve")
    $request.Method = 'POST'
    $request.Proxy = $null
    $request.Timeout = 1800000
    $request.ReadWriteTimeout = 1800000
    $request.ContentType = 'application/json; charset=utf-8'
    $request.Headers.Add('Authorization', "Bearer $($env:AGENT_S_APPROVAL_TOKEN)")
    $request.Headers.Add('X-Agent-S-Agent', "$($env:AGENT_S_AGENT)")
    $bytes = [Text.Encoding]::UTF8.GetBytes($body)
    $request.ContentLength = $bytes.Length
    $stream = $request.GetRequestStream(); $stream.Write($bytes, 0, $bytes.Length); $stream.Close()
    $response = $request.GetResponse()
    $answer = (New-Object IO.StreamReader($response.GetResponseStream())).ReadToEnd()
    $response.Close()
} catch { }
if ($answer -eq 'allow') { exit 0 }
[Console]::Error.WriteLine('The person denied this step. Ask them what to do instead.')
exit 2
