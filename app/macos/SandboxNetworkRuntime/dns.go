package main

import (
	"context"
	"log"
	"net"

	"github.com/miekg/dns"
	"gvisor.dev/gvisor/pkg/tcpip/adapters/gonet"
	"gvisor.dev/gvisor/pkg/tcpip/network/ipv4"
	"gvisor.dev/gvisor/pkg/tcpip/stack"
)

func startDNS(ctx context.Context, network *stack.Stack) error {
	address := virtualAddress(gatewayIP, 53)
	packets, err := gonet.DialUDP(network, &address, nil, ipv4.ProtocolNumber)
	if err != nil {
		return err
	}
	listener, err := gonet.ListenTCP(network, address, ipv4.ProtocolNumber)
	if err != nil {
		_ = packets.Close()
		return err
	}

	handler := dnsGateway{ctx: ctx, queries: make(chan struct{}, 32)}
	servers := []*dns.Server{
		{PacketConn: packets, Handler: handler, UDPSize: 4096},
		{Listener: listener, Handler: handler},
	}
	context.AfterFunc(ctx, func() {
		_ = packets.Close()
		_ = listener.Close()
		for _, server := range servers {
			_ = server.Shutdown()
		}
	})
	for _, server := range servers {
		go func() {
			if err := server.ActivateAndServe(); err != nil && ctx.Err() == nil {
				log.Printf("DNS gateway: %v", err)
			}
		}()
	}
	return nil
}

type dnsGateway struct {
	ctx     context.Context
	queries chan struct{}
}

func (gateway dnsGateway) ServeDNS(writer dns.ResponseWriter, request *dns.Msg) {
	response := new(dns.Msg).SetReply(request)
	if len(request.Question) != 1 {
		response.Rcode = dns.RcodeFormatError
		_ = writer.WriteMsg(response)
		return
	}
	question := request.Question[0]
	if question.Qclass != dns.ClassINET || question.Qtype == dns.TypeAXFR || question.Qtype == dns.TypeIXFR || question.Qtype == dns.TypeANY {
		response.Rcode = dns.RcodeRefused
		_ = writer.WriteMsg(response)
		return
	}
	select {
	case gateway.queries <- struct{}{}:
		defer func() { <-gateway.queries }()
	case <-gateway.ctx.Done():
		return
	default:
		response.Rcode = dns.RcodeServerFailure
		_ = writer.WriteMsg(response)
		return
	}
	response.Answer, response.Rcode = queryHostDNS(question)
	response.RecursionAvailable = true
	if _, udp := writer.LocalAddr().(*net.UDPAddr); udp {
		size := dns.MinMsgSize
		if edns := request.IsEdns0(); edns != nil {
			size = min(int(edns.UDPSize()), 4096)
		}
		response.Truncate(size)
	}
	_ = writer.WriteMsg(response)
}
