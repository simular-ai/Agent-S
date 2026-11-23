import Foundation

/// Tracks one sandbox Codex turn without touching other tools or its saved thread.
public struct SandboxCodexTurn: Sendable {
    public static let lifecycleFailureMarker = "AGENT_S_CODEX_LIFECYCLE:"

    private static let ownerPrefix = "Agent S Codex turn "
    private let unit: String
    private let turnID: UUID

    public init(agentID: UUID, turnID: UUID = UUID()) {
        unit = "agent-s-codex-\(agentID.uuidString.lowercased()).service"
        self.turnID = turnID
    }

    /// Stop any older instance, then stream exactly one invocation under this agent's service.
    public func launchCommand(_ command: String) -> String {
        #"""
        \#(setup)
        if [ -f "$cancelled" ]; then
          lifecycle_failure 'The Codex turn was stopped before it started.'
        fi

        owner=$(current_owner) || lifecycle_failure 'Could not inspect the Codex turn service.'
        case "$owner" in
          '') ;;
          '\#(Self.ownerPrefix)'*) stop_unit ;;
          *) lifecycle_failure 'The Codex turn service has an unexpected owner.' ;;
        esac

        # Check cancellation inside the unit too, in case SSH died during registration.
        exec 8<&0
        systemd-run --user --quiet --wait --pipe --collect --expand-environment=no \
          --service-type=exec --property=KillMode=control-group --property=TimeoutStopSec=10s \
          --property=UnsetEnvironment=DBUS_SESSION_BUS_ADDRESS \
          --unit="$unit" --description="$description" --setenv=PATH="$PATH" \
          /bin/bash -c 'if [ -f "$1" ]; then exit 70; fi; exec /bin/bash -c "$2"' bash \
          "$cancelled" \#(Self.quote(command)) <&8 8<&- 9>&- &
        runner=$!
        exec 8<&-
        # Unlock at registration so a new SSH connection can stop an active turn.
        while kill -0 "$runner" 2>/dev/null; do
          if owner=$(current_owner) && [ "$owner" = "$description" ]; then break; fi
          sleep 0.05
        done
        flock -u 9
        exec 9>&-
        wait "$runner"
        """#
    }

    /// Works over a new SSH connection and cannot stop a newer invocation of the same service.
    public var stopCommand: String {
        #"""
        \#(setup)
        # Also cancel an SSH command that was accepted but has not registered its service yet.
        touch -- "$cancelled" || lifecycle_failure 'Could not mark the Codex turn as stopped.'
        owner=$(current_owner) || lifecycle_failure 'Could not inspect the Codex turn service.'
        if [ "$owner" = "$description" ]; then stop_unit; fi
        """#
    }

    private var setup: String {
        #"""
        unit='\#(unit)'
        description='\#(Self.ownerPrefix)\#(turnID.uuidString)'
        export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
        unset DBUS_SESSION_BUS_ADDRESS
        cancelled="$XDG_RUNTIME_DIR/$unit.\#(turnID.uuidString).cancelled"

        lifecycle_failure() {
          printf '%s %s\n' '\#(Self.lifecycleFailureMarker)' "$1" >&2
          exit 70
        }
        current_owner() {
          local properties
          properties=$(systemctl --user show --property=LoadState --property=Description "$unit" 2>/dev/null)
          case "$properties" in
            *'LoadState=not-found'*) return 0 ;;
            *'LoadState='*) printf '%s\n' "$properties" | sed -n 's/^Description=//p' ;;
            *) return 1 ;;
          esac
        }
        stop_unit() {
          if ! systemctl --user stop "$unit"; then
            owner=$(current_owner) && [ -z "$owner" ] || lifecycle_failure 'Could not stop the previous Codex turn.'
          fi
          # A stopped transient unit can remain loaded briefly while its waiter detaches.
          for ((attempt = 0; attempt < 20; attempt++)); do
            owner=$(current_owner) || lifecycle_failure 'Could not inspect the stopped Codex turn.'
            if [ -z "$owner" ]; then return 0; fi
            sleep 0.05
          done
          lifecycle_failure 'The previous Codex turn is still closing.'
        }

        exec 9>"$XDG_RUNTIME_DIR/$unit.lock" || lifecycle_failure 'Could not open the Codex turn lock.'
        flock -w 15 9 || lifecycle_failure 'Another Codex turn is still starting or stopping.'
        """#
    }

    private static func quote(_ command: String) -> String {
        "'" + command.replacingOccurrences(of: "'", with: "'\\''") + "'"
    }
}
