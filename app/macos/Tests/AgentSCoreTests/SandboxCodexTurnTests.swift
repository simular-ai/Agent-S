import Foundation
import Testing
import AgentSCore

@Suite("Sandbox Codex turn lifetime")
struct SandboxCodexTurnTests {
    private let agentID = UUID(uuidString: "7E896F21-780D-4FB0-9BC2-01209044499E")!
    private let firstID = UUID(uuidString: "4DF0D857-C839-4CFE-956F-01209044499E")!
    private let nextID = UUID(uuidString: "23D706FE-F87C-4DC1-9716-01209044499E")!

    @Test func sameAgentUsesOneServiceWithDistinctTurnOwners() {
        let first = SandboxCodexTurn(agentID: agentID, turnID: firstID)
        let next = SandboxCodexTurn(agentID: agentID, turnID: nextID)
        let unit = "agent-s-codex-\(agentID.uuidString.lowercased()).service"
        for command in [first.launchCommand("true"), first.stopCommand, next.launchCommand("true"), next.stopCommand] {
            #expect(command.contains("unit='\(unit)'"))
            #expect(command.contains("flock -w 15 9"))
            #expect(command.contains(SandboxCodexTurn.lifecycleFailureMarker))
        }
        #expect(first.stopCommand.contains("description='Agent S Codex turn \(firstID.uuidString)'"))
        #expect(next.stopCommand.contains("description='Agent S Codex turn \(nextID.uuidString)'"))
        #expect(first.stopCommand.contains(#"state="$HOME/.agent-s/codex-turns""#))
        #expect(first.stopCommand.contains("Could not stop the previous Codex turn."))
        #expect(first.stopCommand.contains(#"if [ "$owner" = "$description" ]; then stop_unit; fi"#))
        #expect(!first.stopCommand.contains(nextID.uuidString))
    }

    @Test func streamsAndStopsOnlyTheManagedService() {
        let turn = SandboxCodexTurn(agentID: agentID, turnID: firstID)
        let launch = turn.launchCommand(#"codex exec resume thread 'don't rerun $(false)'"#)
        #expect(launch.contains("sudo -n /usr/bin/systemd-run --system --quiet --wait --pipe --collect --expand-environment=no"))
        #expect(launch.contains(#"--uid="$uid" --gid="$gid" --working-directory="$HOME""#))
        #expect(launch.contains(#"uid=$(id -u) && gid=$(id -g) && [ "$uid" -gt 0 ]"#))
        #expect(launch.contains(#"--setenv="HOME=$HOME" --setenv="PATH=$PATH" --setenv="XDG_RUNTIME_DIR=$runtime""#))
        #expect(launch.contains(#"exec 9>"$state/$unit.lock""#))
        #expect(!launch.contains("systemd-run --user"))
        #expect(!launch.contains("systemctl --user"))
        #expect(launch.contains("--property=KillMode=control-group"))
        #expect(launch.contains(#"--property='ExecStopPost=-/bin/sh -c "if [ \"$$SERVICE_RESULT\" = oom-kill ]; then echo; echo AGENT_S_SANDBOX_TURN_OOM; fi >&2"'"#))
        #expect(launch.contains("--property=UnsetEnvironment=DBUS_SESSION_BUS_ADDRESS"))
        #expect(launch.contains(#"exec /bin/bash -c "$2""#))
        #expect(launch.contains(#"9>&- &"#))
        #expect(launch.contains("flock -u 9\nexec 9>&-\nwait"))
        #expect(turn.stopCommand.contains(#"sudo -n /usr/bin/systemctl --system stop "$unit""#))
        #expect(!turn.stopCommand.contains("codex exec"))
        #expect(!turn.stopCommand.contains("pkill"))
    }

    @Test(arguments: [Harness.claudeCode, .deepSeek])
    func otherHarnessesKeepTheirOwnTurnServiceAndStreamingInput(harness: Harness) {
        let turn = SandboxCodexTurn(agentID: agentID, turnID: firstID, harness: harness)
        let unit = "agent-s-\(harness.rawValue)-\(agentID.uuidString.lowercased()).service"
        let launch = turn.launchCommand("cat")
        let stop = turn.stopCommand
        for command in [launch, stop] {
            #expect(command.contains("unit='\(unit)'"))
            #expect(command.contains("description='Agent S \(harness.displayName) turn \(firstID.uuidString)'"))
            #expect(command.contains("state=\"$HOME/.agent-s/\(harness.rawValue)-turns\""))
            #expect(command.contains(SandboxCodexTurn.lifecycleFailureMarker(for: harness)))
            #expect(!command.contains(SandboxCodexTurn.lifecycleFailureMarker))
        }
        #expect(launch.contains("--wait --pipe"))
        #expect(launch.contains("exec 8<&0"))
        #expect(launch.contains("<&8 8<&- 9>&- &"))
        #expect(stop.contains(#"touch -- "$cancelled""#))
        #expect(stop.contains(#"if [ "$owner" = "$description" ]; then stop_unit; fi"#))
        #expect(!stop.contains("systemctl --user"))
    }
}
