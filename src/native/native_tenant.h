#pragma once
// Per-model native tenancy: stages + async mailboxes are keyed by tenant id.
// Thread-local current tenant must be set on the invoking thread before native
// work (Python main or that tenant's async worker). Lookups that run before an
// OMP parallel region see this TLS value.
#include <atomic>
#include <cstdint>

constexpr int32_t NATIVE_MAX_TENANTS = 8;

inline int32_t& native_tls_tenant() {
    thread_local int32_t tenant = 0;
    return tenant;
}

inline int32_t native_tenant_get() {
    return native_tls_tenant();
}

inline void native_tenant_put(int32_t id) {
    if (id < 0 || id >= NATIVE_MAX_TENANTS) {
        id = 0;
    }
    native_tls_tenant() = id;
}

// Tenant 0 is the default (non-contract / unset). 1..MAX-1 are allocated.
inline std::atomic<uint32_t>& native_tenant_alloc_mask() {
    // Bit 0 reserved forever (default tenant).
    static std::atomic<uint32_t> mask{1u};
    return mask;
}
