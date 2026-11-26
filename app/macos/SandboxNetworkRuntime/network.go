package main

import (
	"context"
	"fmt"
	"log"
	"net"
	"net/netip"
	"sync"

	"github.com/containers/gvisor-tap-vsock/pkg/services/dhcp"
	"github.com/containers/gvisor-tap-vsock/pkg/services/forwarder"
	"github.com/containers/gvisor-tap-vsock/pkg/tap"
	"github.com/containers/gvisor-tap-vsock/pkg/types"
	"gvisor.dev/gvisor/pkg/tcpip"
	"gvisor.dev/gvisor/pkg/tcpip/header"
	"gvisor.dev/gvisor/pkg/tcpip/network/arp"
	"gvisor.dev/gvisor/pkg/tcpip/network/ipv4"
	"gvisor.dev/gvisor/pkg/tcpip/stack"
	"gvisor.dev/gvisor/pkg/tcpip/transport/icmp"
	"gvisor.dev/gvisor/pkg/tcpip/transport/tcp"
	"gvisor.dev/gvisor/pkg/tcpip/transport/udp"
)

const (
	mtu        = 1500
	subnetCIDR = "192.168.127.0/24"
	gatewayIP  = "192.168.127.1"
	guestIP    = "192.168.127.2"
	gatewayMAC = "02:00:00:00:01:01"
	nicID      = 1
)

type virtualNetwork struct {
	link  *tap.Switch
	stack *stack.Stack
}

func newVirtualNetwork(ctx context.Context, guestMAC net.HardwareAddr) (*virtualNetwork, error) {
	configuration := &types.Configuration{
		MTU:               mtu,
		Subnet:            subnetCIDR,
		GatewayIP:         gatewayIP,
		GatewayMacAddress: gatewayMAC,
		DHCPStaticLeases:  map[string]string{guestIP: guestMAC.String()},
	}
	pool, err := tap.NewIPPool(subnetCIDR)
	if err != nil {
		return nil, err
	}
	if err := pool.Reserve(gatewayIP, gatewayMAC); err != nil {
		return nil, err
	}
	if err := pool.Reserve(guestIP, guestMAC.String()); err != nil {
		return nil, err
	}
	endpoint, err := tap.NewLinkEndpoint(false, mtu, gatewayMAC, gatewayIP, nil)
	if err != nil {
		return nil, err
	}
	link := tap.NewSwitch(false)
	endpoint.Connect(link)
	link.Connect(endpoint)

	// Use gvisor-tap-vsock's DHCP and forwarding components. The DNS gateway
	// delegates record lookups to macOS so it can use the VPN's scoped DNS.
	s := stack.New(stack.Options{
		NetworkProtocols: []stack.NetworkProtocolFactory{ipv4.NewProtocol, arp.NewProtocol},
		TransportProtocols: []stack.TransportProtocolFactory{
			tcp.NewProtocol,
			udp.NewProtocol,
			icmp.NewProtocol4,
		},
	})
	if err := configureStack(s, endpoint); err != nil {
		s.Close()
		return nil, err
	}
	server, err := dhcp.New(configuration, s, pool)
	if err != nil {
		s.Close()
		return nil, fmt.Errorf("start DHCP server: %w", err)
	}
	if err := startDNS(ctx, s); err != nil {
		_ = server.Underlying.Close()
		s.Close()
		return nil, fmt.Errorf("start DNS gateway: %w", err)
	}
	context.AfterFunc(ctx, func() { _ = server.Underlying.Close() })
	go func() {
		if err := server.Serve(); err != nil && ctx.Err() == nil {
			log.Printf("DHCP server: %v", err)
		}
	}()
	return &virtualNetwork{link: link, stack: s}, nil
}

func configureStack(s *stack.Stack, endpoint *tap.LinkEndpoint) error {
	if err := s.CreateNIC(nicID, endpoint); err != nil {
		return fmt.Errorf("create NIC: %s", err)
	}
	gateway := virtualAddress(gatewayIP, 0).Addr
	if err := s.AddProtocolAddress(nicID, tcpip.ProtocolAddress{
		Protocol:          ipv4.ProtocolNumber,
		AddressWithPrefix: gateway.WithPrefix(),
	}, stack.AddressProperties{}); err != nil {
		return fmt.Errorf("assign gateway address: %s", err)
	}
	if err := s.SetSpoofing(nicID, true); err != nil {
		return fmt.Errorf("configure gateway source routing: %s", err)
	}
	if err := s.SetPromiscuousMode(nicID, true); err != nil {
		return fmt.Errorf("enable gateway forwarding: %s", err)
	}
	prefix := netip.MustParsePrefix(subnetCIDR)
	subnet, err := tcpip.NewSubnet(tcpip.AddrFrom4(prefix.Addr().As4()), tcpip.MaskFromBytes(net.CIDRMask(prefix.Bits(), 32)))
	if err != nil {
		return fmt.Errorf("create guest subnet: %w", err)
	}
	s.SetRouteTable([]tcpip.Route{{Destination: subnet, NIC: nicID}})

	var natLock sync.Mutex
	tcpForwarder := forwarder.TCP(s, nil, &natLock, false, 0, 0)
	udpForwarder := forwarder.UDP(s, nil, &natLock, false)
	icmpForwarder := forwarder.ICMP(s, nil, &natLock)
	for protocol, forward := range map[tcpip.TransportProtocolNumber]func(stack.TransportEndpointID, *stack.PacketBuffer) bool{
		tcp.ProtocolNumber:   tcpForwarder.HandlePacket,
		udp.ProtocolNumber:   udpForwarder.HandlePacket,
		icmp.ProtocolNumber4: icmpForwarder.HandlePacket,
	} {
		s.SetTransportProtocolHandler(protocol, func(id stack.TransportEndpointID, packet *stack.PacketBuffer) bool {
			destination := id.LocalAddress
			// DNS and DHCP already have endpoints on the virtual network. Never
			// translate other traffic to the gateway or host's loopback sockets.
			if destination == gateway || destination == header.IPv4Any || header.IsV4LoopbackAddress(destination) {
				return false
			}
			return forward(id, packet)
		})
	}
	return nil
}

func virtualAddress(ip string, port uint16) tcpip.FullAddress {
	return tcpip.FullAddress{NIC: nicID, Addr: tcpip.AddrFrom4(netip.MustParseAddr(ip).As4()), Port: port}
}
