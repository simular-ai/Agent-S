import Darwin
import Foundation

/// An estimate of available RAM from free and inactive pages.
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
        // The kernel's free count includes speculative pages.
        let pages = UInt64(stats.free_count) + UInt64(stats.inactive_count)
        return min(pages * UInt64(vm_kernel_page_size), total)
    }

    /// "6 GB", "9.6 GB".
    static func format(_ bytes: UInt64) -> String {
        let gb = Double(bytes) / 1_073_741_824
        return gb >= 10 || gb.rounded() == gb ? "\(Int(gb.rounded())) GB" : String(format: "%.1f GB", gb)
    }
}
