package main

/*
#include <dns_sd.h>
#include <errno.h>
#include <poll.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_RECORDS 32
#define MAX_RDATA 4096
#define TIMEOUT_MS 5000

struct dns_record {
	char name[kDNSServiceMaxDomainName];
	uint16_t type, class, length;
	uint32_t ttl;
	unsigned char data[MAX_RDATA];
};

struct dns_result {
	struct dns_record records[MAX_RECORDS];
	uint16_t count, requested_type;
	int done;
	DNSServiceErrorType error;
};

static void receive_record(DNSServiceRef ref, DNSServiceFlags flags, uint32_t interface,
	DNSServiceErrorType error, const char *name, uint16_t type, uint16_t class,
	uint16_t length, const void *data, uint32_t ttl, void *context) {
	struct dns_result *result = context;
	if (error != kDNSServiceErr_NoError) {
		result->error = error;
		result->done = 1;
		return;
	}
	if ((flags & kDNSServiceFlagsAdd) == 0) {
		return;
	}
	if (result->count == MAX_RECORDS || length > MAX_RDATA || strlen(name) >= kDNSServiceMaxDomainName) {
		result->error = kDNSServiceErr_Unknown;
		result->done = 1;
		return;
	}
	struct dns_record *record = &result->records[result->count++];
	strlcpy(record->name, name, sizeof(record->name));
	record->type = type;
	record->class = class;
	record->ttl = ttl;
	record->length = length;
	memcpy(record->data, data, length);
	if ((flags & kDNSServiceFlagsMoreComing) == 0 && type == result->requested_type) {
		result->done = 1;
	}
}

static int64_t monotonic_ms(void) {
	struct timespec now;
	clock_gettime(CLOCK_MONOTONIC, &now);
	return (int64_t)now.tv_sec * 1000 + now.tv_nsec / 1000000;
}

static DNSServiceErrorType query_record(const char *name, uint16_t type, uint16_t class, struct dns_result *result) {
	DNSServiceRef service;
	result->requested_type = type;
	DNSServiceErrorType error = DNSServiceQueryRecord(&service, kDNSServiceFlagsReturnIntermediates,
		kDNSServiceInterfaceIndexAny, name, type, class, receive_record, result);
	if (error != kDNSServiceErr_NoError) {
		return error;
	}
	int64_t deadline = monotonic_ms() + TIMEOUT_MS;
	struct pollfd event = { .fd = DNSServiceRefSockFD(service), .events = POLLIN };
	while (!result->done) {
		int remaining = (int)(deadline - monotonic_ms());
		if (remaining <= 0) {
			break;
		}
		int ready = poll(&event, 1, remaining);
		if (ready < 0 && errno == EINTR) {
			continue;
		}
		if (ready == 0) {
			break;
		}
		if (ready < 0 || (event.revents & POLLIN) == 0) {
			error = kDNSServiceErr_Unknown;
			break;
		}
		error = DNSServiceProcessResult(service);
		if (error != kDNSServiceErr_NoError) {
			break;
		}
	}
	DNSServiceRefDeallocate(service);
	if (error != kDNSServiceErr_NoError) {
		return error;
	}
	if (!result->done && result->count == 0) {
		return kDNSServiceErr_Timeout;
	}
	return result->error;
}
*/
import "C"

import (
	"unsafe"

	"github.com/miekg/dns"
)

func queryHostDNS(question dns.Question) ([]dns.RR, int) {
	name := C.CString(question.Name)
	defer C.free(unsafe.Pointer(name))
	var result C.struct_dns_result
	switch C.query_record(name, C.uint16_t(question.Qtype), C.uint16_t(question.Qclass), &result) {
	case C.kDNSServiceErr_NoError, C.kDNSServiceErr_NoSuchRecord:
		// macOS combines NXDOMAIN and empty record sets in NoSuchRecord.
		// Respond with an empty answer in either case.
	default:
		return nil, dns.RcodeServerFailure
	}

	answers := make([]dns.RR, 0, result.count)
	for index := range int(result.count) {
		record := &result.records[index]
		header := dns.RR_Header{
			Name:     C.GoString(&record.name[0]),
			Rrtype:   uint16(record._type),
			Class:    uint16(record.class),
			Ttl:      uint32(record.ttl),
			Rdlength: uint16(record.length),
		}
		data := C.GoBytes(unsafe.Pointer(&record.data[0]), C.int(record.length))
		answer, _, err := dns.UnpackRRWithHeader(header, data, 0)
		if err != nil {
			return nil, dns.RcodeServerFailure
		}
		answers = append(answers, answer)
	}
	return answers, dns.RcodeSuccess
}
