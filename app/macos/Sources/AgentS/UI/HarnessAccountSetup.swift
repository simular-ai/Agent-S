import SwiftUI
import AgentSCore

/// Sign-in for an account-backed agent, shown before chatting and in Agent Settings.
struct HarnessAccountSetup: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            if let notice = model.provider.accountDataNotice {
                Text(notice).font(.caption).foregroundStyle(.secondary)
            }
            accountControls
        }
    }

    @ViewBuilder private var accountControls: some View {
        switch model.accountStatus {
        case .unchecked, .checking:
            ProgressView("Checking sign-in…").controlSize(.small)
        case .signedOut:
            Button(model.provider == .claudeAccount ? "Sign in with Claude Code…" : "Sign in with ChatGPT…", action: signIn)
        case .connecting:
            ProgressView("Waiting for sign-in…").controlSize(.small)
            Button("Cancel sign-in") { model.cancelAccountSignIn() }
        case .awaitingSignIn:
            Text("Finish signing in through Claude Code in Terminal.").font(.callout)
            HStack {
                Button("Check sign-in", action: checkSignIn)
                Button("Reopen sign-in…", action: signIn)
            }
        case .connected:
            Label("Signed in", systemImage: "checkmark.circle.fill").foregroundStyle(.green)
        case let .failed(message):
            Text(message).font(.callout).foregroundStyle(.red)
            Button("Check sign-in", action: checkSignIn)
        }
    }

    private func checkSignIn() {
        Task { await model.refreshAccountStatus() }
    }

    private func signIn() {
        Task { await model.signInToAccount() }
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
