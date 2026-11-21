using System.Net;
using System.Text;
using System.Text.Json;
using AgentS.Core;

var passed = 0;
void Check(bool condition, string description)
{
    if (!condition) throw new Exception("FAIL: " + description);
    passed++; Console.WriteLine("PASS: " + description);
}
async Task<List<JsonElement>> Read(string value)
{
    var entries = new List<JsonElement>();
    await foreach (var entry in EventStream.Read(new StringReader(value))) entries.Add(entry);
    return entries;
}
var events = await Read(": keepalive\n\ndata: {\"type\":\n" + "data: \"finish\"}\n\ndata: [DONE]\n\n");
Check(events.Count == 1 && events[0].GetProperty("type").GetString() == "finish", "SSE comments, multiline data, blank frames and DONE");
Check((await Read("data: {\"type\":\"finish\"}")).Count == 1, "SSE final frame without trailing blank line");
try { await Read("data: invalid\n\n"); throw new Exception("Malformed event was accepted"); }
catch (JsonException) { Check(true, "Malformed stream fails visibly"); }
try { await Read("data: " + new string('a', 1_048_577) + "\n\n"); throw new Exception("Oversized event was accepted"); }
catch (SaiException) { Check(true, "Oversized event rejected"); }
Check(SaiClient.SafeWebLink("https://example.test/approval") is not null, "Explicit HTTPS approval allowed");
foreach (var unsafeUrl in new[] { "file:///tmp/file", "javascript:alert(1)", "http://example.test", "https://user:secret@example.test/" })
    Check(SaiClient.SafeWebLink(unsafeUrl) is null, "Unsafe approval link refused: " + unsafeUrl.Split(':')[0]);
Check(new Computer("c", "Computer", "cloud", "windows", true, "hibernated", true).Availability == "Asleep", "Wakeable online cloud computer is not shown as running");
Check(new Computer("c", "Computer", "own", "windows", false, "active", false).Availability == "Offline", "Enrolled offline computer remains offline");

var handler = new RecordingHandler();
using var client = new SaiClient("test-key", handler);
handler.Reply = "{\"machines\":[{\"machineId\":\"pc1\",\"name\":\"PC\",\"kind\":\"own\",\"online\":true,\"canWake\":false}]}";
Check((await client.Computers())[0].Id == "pc1", "Computer metadata uses existing Swift contract");
Check(handler.Path == "/v1/agents/machines" && handler.Auth == "Bearer test-key", "Authenticated machine discovery contract");
handler.Reply = "{\"models\":[{\"id\":\"auto\",\"name\":\"Auto\",\"allowed\":true}]}";
Check((await client.Models())[0].Allowed, "Model eligibility read from API");
handler.Sse = true; handler.Reply = "data: {\"type\":\"finish\"}\n\ndata: [DONE]\n\n";
await foreach (var entry in client.Run("hello", "pc1", "session1", "auto")) { }
using (var body = JsonDocument.Parse(handler.Body!))
{
    Check(handler.Path == "/v1/agents/message" && handler.Method == HttpMethod.Post, "POST to existing message endpoint");
    Check(body.RootElement.GetProperty("sessionId").GetString() == "session1" && !body.RootElement.TryGetProperty("model", out _), "Resume session and omit automatic model override");
}
await foreach (var entry in client.Run("hello", "pc1", null, "specific-model")) { }
using (var body = JsonDocument.Parse(handler.Body!))
    Check(!body.RootElement.TryGetProperty("sessionId", out _) && body.RootElement.GetProperty("model").GetString() == "specific-model", "New conversation omits session and sends explicit model");
handler.Reply = "data: {\"type\":\"text-delta\",\"delta\":\"partial\"}\n\n";
try { await foreach (var entry in client.Run("hello", "pc1", null, null)) { } throw new Exception("Early EOF accepted"); }
catch (SaiException error) { Check(error.Message.Contains("before confirming"), "Early EOF is not reported as completed work"); }
handler.Sse = false; handler.Reply = "{}";
await client.Approve("a1", "deny");
Check(handler.Path == "/v1/agents/approve" && handler.Body!.Contains("\"response\":\"deny\""), "Approval API preserves denial semantics");
await client.Abort("s1");
Check(handler.Path == "/v1/agents/abort" && handler.Body!.Contains("\"sessionId\":\"s1\""), "Stop is an explicit server abort, not just closing the stream");
handler.Reply = "{\"waking\":true}"; await client.Wake("pc1");
Check(handler.Path == "/v1/agents/wake", "Wake contract");
handler.Reply = "{\"canWake\":false}";
try { await client.Wake("pc1"); throw new Exception("Unconfirmed wake accepted"); }
catch (SaiException) { Check(true, "Unconfirmed wake reported as failure"); }
handler.Reply = "{\"liveUrl\":\"https://example.test/screen?token=private\"}";
await client.LiveLink("computer/with?reserved");
Check(handler.Path!.Contains("computer%2Fwith%3Freserved"), "Machine ID is escaped as one path segment");
handler.Status = HttpStatusCode.Unauthorized; handler.Reply = "secret response-body";
try { await client.Computers(); throw new Exception("Unauthorized accepted"); }
catch (SaiException error) { Check(error.Message.Contains("rejected") && !error.Message.Contains("secret"), "Auth failure is visible and body is not leaked"); }
handler.Status = HttpStatusCode.Redirect;
try { await client.Abort("s1"); throw new Exception("Redirect accepted"); }
catch (SaiException) { Check(true, "Non-success redirect not treated as confirmed abort"); }
var saved = JsonSerializer.Serialize(new Conversation { Entries = [new ChatEntry { Text = "Approval", ApprovalId = "id", ApprovalUrl = new Uri("https://example.test/private-token") }] });
Check(!saved.Contains("private-token") && !saved.Contains("ApprovalId"), "Approval URLs and actionable IDs are excluded from local history");
Console.WriteLine($"{passed} assertions passed.");

sealed class RecordingHandler : HttpMessageHandler
{
    public string Reply = "{}";
    public HttpStatusCode Status = HttpStatusCode.OK;
    public bool Sse;
    public string? Body, Path, Auth;
    public HttpMethod? Method;
    protected override async Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
    {
        Body = request.Content is null ? null : await request.Content.ReadAsStringAsync(cancellationToken);
        Path = request.RequestUri!.AbsolutePath;
        Auth = request.Headers.Authorization?.ToString(); Method = request.Method;
        return new HttpResponseMessage(Status) { Content = new StringContent(Reply, Encoding.UTF8, Sse ? "text/event-stream" : "application/json") };
    }
}
