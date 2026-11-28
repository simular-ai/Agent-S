import Foundation

/// Tracks a sandbox harness turn without touching the desktop or saved conversation.
public struct SandboxCodexTurn: Sendable {
    public static let lifecycleFailureMarker = "AGENT_S_CODEX_LIFECYCLE:"

    public static func lifecycleFailureMarker(for harness: Harness) -> String {
        harness == .codex ? lifecycleFailureMarker : "AGENT_S_SANDBOX_TURN_LIFECYCLE:"
    }

    private let harness: Harness
    private let unit: String
    private let turnID: UUID
    private var ownerPrefix: String { "Agent S \(harness.displayName) turn " }

    public init(agentID: UUID, turnID: UUID = UUID(), harness: Harness = .codex) {
        self.harness = harness
        unit = "agent-s-\(harness.rawValue)-\(agentID.uuidString.lowercased()).service"
        self.turnID = turnID
    }

    /// Stop any older instance, then stream exactly one invocation under this agent's service.
    public func launchCommand(_ command: String) -> String {
        #"""
        \#(setup)
        if [ -f "$cancelled" ]; then
          lifecycle_failure 'The \#(harness.displayName) turn was stopped before it started.'
        fi

        owner=$(current_owner) || lifecycle_failure 'Could not inspect the \#(harness.displayName) turn service.'
        case "$owner" in
          '') ;;
          '\#(ownerPrefix)'*) stop_unit ;;
          *) lifecycle_failure 'The \#(harness.displayName) turn service has an unexpected owner.' ;;
        esac

        # Use the guest's system manager because streaming needs a D-Bus socket.
        # The transient service still runs with the SSH user's identity and profile.
        # Report the service's OOM result even when the harness cannot write a final event.
        exec 8<&0
        sudo -n /usr/bin/systemd-run --system --quiet --wait --pipe --collect --expand-environment=no \
          --uid="$uid" --gid="$gid" --working-directory="$HOME" \
          --setenv="HOME=$HOME" --setenv="PATH=$PATH" --setenv="XDG_RUNTIME_DIR=$runtime" \
          --service-type=exec --property=KillMode=control-group --property=TimeoutStopSec=10s \
          --property=UnsetEnvironment=DBUS_SESSION_BUS_ADDRESS \
          --property='ExecStopPost=-/bin/sh -c "if [ \"$$SERVICE_RESULT\" = oom-kill ]; then echo; echo \#(SandboxTurnFailure.outOfMemoryMarker); fi >&2"' \
          --unit="$unit" --description="$description" -- \
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
        touch -- "$cancelled" || lifecycle_failure 'Could not mark the \#(harness.displayName) turn as stopped.'
        owner=$(current_owner) || lifecycle_failure 'Could not inspect the \#(harness.displayName) turn service.'
        if [ "$owner" = "$description" ]; then stop_unit; fi
        """#
    }

    private var setup: String {
        #"""
        unit='\#(unit)'
        description='\#(ownerPrefix)\#(turnID.uuidString)'
        unset DBUS_SESSION_BUS_ADDRESS

        lifecycle_failure() {
          printf '%s %s\n' '\#(Self.lifecycleFailureMarker(for: harness))' "$1" >&2
          exit 70
        }
        current_owner() {
          local properties
          properties=$(sudo -n /usr/bin/systemctl --system show --property=LoadState --property=Description "$unit" 2>/dev/null)
          case "$properties" in
            *'LoadState=not-found'*) return 0 ;;
            *'LoadState='*) printf '%s\n' "$properties" | sed -n 's/^Description=//p' ;;
            *) return 1 ;;
          esac
        }
        stop_unit() {
          if ! sudo -n /usr/bin/systemctl --system stop "$unit"; then
            owner=$(current_owner) && [ -z "$owner" ] || lifecycle_failure 'Could not stop the previous \#(harness.displayName) turn.'
          fi
          # A stopped transient unit can remain loaded briefly while its waiter detaches.
          for ((attempt = 0; attempt < 20; attempt++)); do
            owner=$(current_owner) || lifecycle_failure 'Could not inspect the stopped \#(harness.displayName) turn.'
            if [ -z "$owner" ]; then return 0; fi
            sleep 0.05
          done
          lifecycle_failure 'The previous \#(harness.displayName) turn is still closing.'
        }

        uid=$(id -u) && gid=$(id -g) && [ "$uid" -gt 0 ] || lifecycle_failure 'Could not identify the sandbox user.'
        runtime="${XDG_RUNTIME_DIR:-/run/user/$uid}"
        state="$HOME/.agent-s/\#(harness.rawValue)-turns"
        (umask 077 && mkdir -p -- "$state") || lifecycle_failure 'Could not create the \#(harness.displayName) turn directory.'
        umask 077
        cancelled="$state/$unit.\#(turnID.uuidString).cancelled"
        exec 9>"$state/$unit.lock" || lifecycle_failure 'Could not open the \#(harness.displayName) turn lock.'
        flock -w 15 9 || lifecycle_failure 'Another \#(harness.displayName) turn is still starting or stopping.'
        """#
    }

    private static func quote(_ command: String) -> String {
        "'" + command.replacingOccurrences(of: "'", with: "'\\''") + "'"
    }
}
