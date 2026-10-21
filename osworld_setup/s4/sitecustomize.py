"""Child-Python hook for the opt-in test network guard."""
import os

if os.environ.get("SHADOW_EVOL_TEST_DENY_NETWORK") == "1":
    from shadow_evol_test_network_guard import install_network_guard

    install_network_guard()
