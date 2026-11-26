package main

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/containers/gvisor-tap-vsock/pkg/types"
	"gvisor.dev/gvisor/pkg/tcpip/adapters/gonet"
	"gvisor.dev/gvisor/pkg/tcpip/header"
	"gvisor.dev/gvisor/pkg/tcpip/network/ipv4"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, "sandbox-network:", err)
		os.Exit(1)
	}
}

func run() error {
	if len(os.Args) != 2 {
		return fmt.Errorf("usage: %s guest-mac-address", os.Args[0])
	}
	mac, err := net.ParseMAC(os.Args[1])
	if err != nil || len(mac) != 6 || mac[0]&1 != 0 || mac.String() == gatewayMAC {
		return fmt.Errorf("invalid guest MAC address %q", os.Args[1])
	}
	parent := os.Getppid()
	if parent <= 1 {
		return fmt.Errorf("parent application is unavailable")
	}
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGTERM, syscall.SIGINT)
	defer stop()
	go watchParent(ctx, parent, stop)

	ethernet, err := net.FileConn(os.Stdin)
	if err != nil {
		return fmt.Errorf("open VM Ethernet socket: %w", err)
	}
	defer ethernet.Close()
	_ = os.Stdin.Close()
	if ethernet.LocalAddr().Network() != "unixgram" || ethernet.RemoteAddr() == nil {
		return fmt.Errorf("VM Ethernet socket must be a connected Unix datagram socket")
	}

	network, err := newVirtualNetwork(ctx, mac)
	if err != nil {
		return err
	}
	defer network.stack.Close()
	listener, err := net.ListenTCP("tcp4", &net.TCPAddr{IP: net.IPv4(127, 0, 0, 1)})
	if err != nil {
		return fmt.Errorf("start SSH proxy: %w", err)
	}
	defer listener.Close()
	context.AfterFunc(ctx, func() {
		_ = ethernet.Close()
		_ = listener.Close()
	})
	go network.serveSSH(ctx, listener)

	if _, err := fmt.Fprintln(os.Stdout, listener.Addr().(*net.TCPAddr).Port); err != nil {
		return fmt.Errorf("report SSH proxy port: %w", err)
	}
	_ = os.Stdout.Close()

	err = network.link.Accept(ctx, ethernetConn{Conn: ethernet, guestMAC: mac}, types.BessProtocol)
	if ctx.Err() != nil {
		return nil
	}
	return err
}

// A Unix datagram socket does not report EOF when the application exits.
func watchParent(ctx context.Context, parent int, stop context.CancelFunc) {
	ticker := time.NewTicker(time.Second)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			if os.Getppid() != parent {
				stop()
				return
			}
		}
	}
}

type ethernetConn struct {
	net.Conn
	guestMAC net.HardwareAddr
}

func (c ethernetConn) Read(packet []byte) (int, error) {
	for {
		n, err := c.Conn.Read(packet)
		if err != nil {
			return 0, err
		}
		if n >= header.EthernetMinimumSize && bytes.Equal(packet[6:12], c.guestMAC) {
			return n, nil
		}
	}
}

func (network *virtualNetwork) serveSSH(ctx context.Context, listener *net.TCPListener) {
	for {
		client, err := listener.AcceptTCP()
		if err != nil {
			return
		}
		go network.proxySSH(ctx, client)
	}
}

func (network *virtualNetwork) proxySSH(ctx context.Context, client *net.TCPConn) {
	defer client.Close()
	dialContext, cancel := context.WithTimeout(ctx, 5*time.Second)
	guest, err := gonet.DialContextTCP(dialContext, network.stack, virtualAddress(guestIP, 22), ipv4.ProtocolNumber)
	cancel()
	if err != nil {
		return
	}
	defer guest.Close()
	stop := context.AfterFunc(ctx, func() {
		_ = client.Close()
		_ = guest.Close()
	})
	defer stop()

	done := make(chan struct{})
	go func() {
		_, _ = io.Copy(guest, client)
		_ = guest.CloseWrite()
		close(done)
	}()
	_, _ = io.Copy(client, guest)
	_ = client.CloseWrite()
	<-done
}
