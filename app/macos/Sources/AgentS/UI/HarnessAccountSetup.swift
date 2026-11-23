import AppKit
import SwiftUI
import AgentSCore

/// Sign-in for an account-backed agent, shown before chatting and in Agent Settings.
struct HarnessAccountSetup: View {
    @EnvironmentObject private var model: AppModel
    @Environment(\.openURL) private var openURL
    var onSandboxSignIn: () -> Void = {}

    private var inSandbox: Bool { model.privacy.usesLocalComputer }
    private var signInTitle: String {
        model.provider == .claudeAccount ? "Sign in with Claude Code…" : "Sign in with ChatGPT…"
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            if let notice = model.provider.accountDataNotice {
                Text(notice).font(.caption).foregroundStyle(.secondary)
            }
            if inSandbox {
                Text("Agents using this sandbox can access its sign-in.").font(.caption).foregroundStyle(.secondary)
            }
            if !model.accountComputerReady {
                Text("Turn on the sandbox to sign in.").font(.callout)
            } else {
                accountControls
            }
        }
    }

    @ViewBuilder private var accountControls: some View {
        switch model.accountStatus {
        case .unchecked, .checking:
            ProgressView("Checking sign-in…").controlSize(.small)
        case .signedOut:
            signInButton(signInTitle)
        case .connecting:
            let sandboxCodex = inSandbox && model.provider == .codexAccount
            ProgressView(sandboxCodex ? "Preparing sign-in…" : inSandbox ? "Opening sign-in…" : "Waiting for sign-in…")
                .controlSize(.small)
            if !inSandbox || sandboxCodex { Button("Cancel sign-in") { model.cancelAccountSignIn() } }
        case let .awaitingDeviceCode(code):
            deviceCodeControls(code)
        case .awaitingSignIn:
            Text(inSandbox
                 ? "Use Take over in the Computer panel to finish signing in. Then check sign-in here."
                 : "Finish signing in through Claude Code in Terminal.")
                .font(.callout)
            HStack {
                Button("Check sign-in", action: checkSignIn)
                signInButton("Reopen sign-in…")
            }
        case .connected:
            Label("Signed in", systemImage: "checkmark.circle.fill").foregroundStyle(.green)
        case let .failed(message):
            Text(message).font(.callout).foregroundStyle(.red)
            HStack {
                Button("Check sign-in", action: checkSignIn)
                if inSandbox { signInButton(signInTitle) }
            }
        }
    }

    private func deviceCodeControls(_ code: CodexAccount.DeviceCode) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Enter this device code in your browser.").font(.callout)
            Text(verbatim: code.userCode)
                .font(.system(.title3, design: .monospaced).weight(.semibold))
                .textSelection(.enabled)
                .accessibilityLabel("Device code \(code.userCode)")
            HStack(spacing: 8) {
                Button("Open browser") { openURL(code.verificationURL) }
                Button("Copy code") {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(code.userCode, forType: .string)
                }
                Button("Cancel", role: .cancel) { model.cancelAccountSignIn() }
                    .accessibilityLabel("Cancel sign-in")
            }
            ProgressView("Waiting for sign-in…").controlSize(.small)
        }
    }

    private func signInButton(_ title: String) -> some View {
        Button(title, action: signIn)
            .disabled(inSandbox && model.sandboxHasActiveTurn)
            .help(inSandbox && model.sandboxHasActiveTurn ? "Wait for the sandbox task to finish" : "")
    }

    private func checkSignIn() {
        Task { await model.refreshAccountStatus() }
    }

    private func signIn() {
        let agentID = model.agent.id
        let provider = model.provider
        let usesSandbox = inSandbox
        Task {
            guard model.agent.id == agentID, model.provider == provider else { return }
            await model.signInToAccount()
            if usesSandbox, model.agent.id == agentID, model.provider == provider, case .awaitingSignIn = model.accountStatus {
                onSandboxSignIn()
            }
        }
    }
}

extension ModelProvider {
    var accountDataNotice: String? {
        let company: String
        switch self {
        case .claudeAccount: company = "Anthropic"
        case .codexAccount: company = "OpenAI"
        default: return nil
        }
        return "Prompts and tool results, including screenshots, go to \(company)."
    }
}
