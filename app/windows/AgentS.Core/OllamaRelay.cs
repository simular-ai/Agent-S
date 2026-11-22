using System.Buffers.Binary;
using System.Collections.Concurrent;

namespace AgentS.Core;

/// <summary>
/// The host half of the guest's Ollama relay. A WSL guest can't reach the host's 127.0.0.1, so a small
/// listener in the guest (agent-s-ollama-relay) accepts connections on its own 127.0.0.1:11434 and
/// multiplexes them over its stdio, which reaches the app through wsl.exe. Each frame is
/// type (1 byte), connection id (uint32, big-endian), length (uint32, big-endian), payload.
/// The guest opens and sends; the host connects each one to Ollama and sends back.
/// </summary>
public sealed class OllamaRelay
{
    public const byte Open = 1, Data = 2, Close = 3;
    public const int MaxPayload = 1 << 20;

    private readonly Stream fromGuest, toGuest;
    private readonly Func<int, CancellationToken, Task<Stream>> connect;
    private readonly ConcurrentDictionary<uint, Stream> connections = new();
    private readonly SemaphoreSlim writeGate = new(1, 1);

    /// <param name="connect">
    /// Opens a host connection for the guest port the connection arrived on: 11434 for Ollama, 11435 for the
    /// app's OpenRouter proxy. An open frame carries that port (uint16, big-endian); an empty one means 11434.
    /// </param>
    public OllamaRelay(Stream fromGuest, Stream toGuest, Func<int, CancellationToken, Task<Stream>> connect)
    {
        this.fromGuest = fromGuest; this.toGuest = toGuest; this.connect = connect;
    }

    public OllamaRelay(Stream fromGuest, Stream toGuest, Func<CancellationToken, Task<Stream>> connect)
        : this(fromGuest, toGuest, (_, token) => connect(token)) { }

    /// <summary>Relays until the guest closes its side or <paramref name="cancellation"/> fires.</summary>
    public async Task Run(CancellationToken cancellation)
    {
        var header = new byte[9];
        try
        {
            while (await ReadExactly(fromGuest, header, cancellation))
            {
                byte type = header[0];
                uint id = BinaryPrimitives.ReadUInt32BigEndian(header.AsSpan(1));
                int length = checked((int)BinaryPrimitives.ReadUInt32BigEndian(header.AsSpan(5)));
                if (length > MaxPayload) throw new InvalidDataException("Relay frame is too large.");
                var payload = new byte[length];
                if (length > 0 && !await ReadExactly(fromGuest, payload, cancellation)) break;
                switch (type)
                {
                    case Open:
                        try
                        {
                            int port = length >= 2 ? BinaryPrimitives.ReadUInt16BigEndian(payload) : 11434;
                            var stream = await connect(port, cancellation);
                            connections[id] = stream;
                            _ = Pump(id, stream, cancellation);
                        }
                        catch (Exception) when (!cancellation.IsCancellationRequested) { await Send(Close, id, default, cancellation); }
                        break;
                    case Data when connections.TryGetValue(id, out var target):
                        try { await target.WriteAsync(payload, cancellation); await target.FlushAsync(cancellation); }
                        catch (Exception) when (!cancellation.IsCancellationRequested) { Drop(id); await Send(Close, id, default, cancellation); }
                        break;
                    case Close:
                        Drop(id);
                        break;
                }
            }
        }
        finally
        {
            foreach (var id in connections.Keys) Drop(id);
        }
    }

    private async Task Pump(uint id, Stream stream, CancellationToken cancellation)
    {
        var buffer = new byte[64 * 1024];
        try
        {
            int read;
            while ((read = await stream.ReadAsync(buffer, cancellation)) > 0)
                await Send(Data, id, buffer.AsMemory(0, read), cancellation);
        }
        catch (Exception) when (!cancellation.IsCancellationRequested) { }
        catch (OperationCanceledException) { return; }
        if (connections.ContainsKey(id))
        {
            Drop(id);
            try { await Send(Close, id, default, cancellation); } catch (Exception) { }
        }
    }

    private void Drop(uint id)
    {
        if (connections.TryRemove(id, out var stream)) stream.Dispose();
    }

    private async Task Send(byte type, uint id, ReadOnlyMemory<byte> payload, CancellationToken cancellation)
    {
        var header = new byte[9];
        header[0] = type;
        BinaryPrimitives.WriteUInt32BigEndian(header.AsSpan(1), id);
        BinaryPrimitives.WriteUInt32BigEndian(header.AsSpan(5), (uint)payload.Length);
        await writeGate.WaitAsync(cancellation);
        try
        {
            await toGuest.WriteAsync(header, cancellation);
            if (payload.Length > 0) await toGuest.WriteAsync(payload, cancellation);
            await toGuest.FlushAsync(cancellation);
        }
        finally { writeGate.Release(); }
    }

    private static async Task<bool> ReadExactly(Stream stream, byte[] buffer, CancellationToken cancellation)
    {
        int done = 0;
        while (done < buffer.Length)
        {
            int read = await stream.ReadAsync(buffer.AsMemory(done), cancellation);
            if (read == 0) return done == 0 ? false : throw new EndOfStreamException("The relay closed mid-frame.");
            done += read;
        }
        return true;
    }
}
