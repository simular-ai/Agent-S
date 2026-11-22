import Darwin
import Foundation

/// How much memory the Mac can give right now: free pages plus the ones macOS would hand over
/// without swapping (inactive, purgeable and speculative), the way Activity Monitor thinks of
/// "available".
enum SystemMemory {
    static var total: UInt64 { ProcessInfo.processInfo.physicalMemory }

    static func available() -> UInt64? {
        var stats = vm_statistics64()
        var count = mach_msg_type_number_t(MemoryLayout<vm_statistics64_data_t>.size / MemoryLayout<integer_t>.size)
        let result = withUnsafeMutablePointer(to: &stats) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &count)
            }
        }
        guard result == KERN_SUCCESS else { return nil }
        let pages = UInt64(stats.free_count) + UInt64(stats.inactive_count)
            + UInt64(stats.purgeable_count) + UInt64(stats.speculative_count)
        return pages * UInt64(vm_kernel_page_size)
    }

    /// "6 GB", "9.6 GB".
    static func format(_ bytes: UInt64) -> String {
        let gb = Double(bytes) / 1_073_741_824
        return gb >= 10 || gb.rounded() == gb ? "\(Int(gb.rounded())) GB" : String(format: "%.1f GB", gb)
    }
}
