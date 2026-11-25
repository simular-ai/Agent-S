import Foundation
import Testing
@testable import AgentSCore

@Suite("Public Ollama model library")
struct OllamaCatalogTests {
    @Test func preservesTheEntireLibraryOrder() throws {
        let models = ["nemotron-3-nano", "gemma4", "qwen3.5"] + (1...25).map { "model-\($0)" }
        let links = models.map { "<li><a href=\"/library/\($0)\">\($0)</a></li>" }.joined(separator: "\n")
        let html = """
        <nav><a href="/library/featured">Featured</a></nav>
        <div id="repo">
            <ul role="list">
                \(links)
                <li><a href="/library/gemma4">Duplicate</a></li>
                <li><a href="/library/gemma4:latest">Tag, not family</a></li>
                <li><a href="https://example.com/library/model">External</a></li>
            </ul>
        </div>
        """
        #expect(try OllamaCatalog.parseFamilies(html) == models)
    }

    @Test func acceptsOnlyOneExactFamilyPathSegment() throws {
        #expect(try OllamaCatalog.tagsURL(for: "qwen3.5").absoluteString == "https://ollama.com/library/qwen3.5/tags")
        for invalid in ["", "../gemma4", "owner/model", "gemma4:4b", " qwen3.5", "gemma4%2Ftags", "gemma4?x=1"] {
            #expect(throws: OllamaCatalog.Failure.invalidFamily) { try OllamaCatalog.tagsURL(for: invalid) }
        }
    }

    @Test func preservesExactTagsAndReadsOnlyLocalDownloadSizes() throws {
        let html = """
        <p class="block col-span-6 md:hidden">4 models</p>
        <a href="/library/gemma4" class="md:hidden flex group"><svg title="Cloud"></svg>Medium Usage</a>
        <a href="/library/gemma4:4b" class="md:hidden flex group">
          <span>28e2e9e36b77</span> • 2.8GB - 2.9GB • 256K context window
        </a>
        <a href="/library/gemma4:4b" class="group-hover:underline">gemma4:4b</a>
        <a href="/library/gemma4:4b-q4_K_M" class="md:hidden flex group"><span>28e2e9e36b77</span> • 420MB • 256K context window</a>
        <a href="/library/gemma4:large-cloud" class="md:hidden flex group">Medium Usage</a>
        """
        let models = try OllamaCatalog.parseVariants(html, family: "gemma4")
        #expect(models.map(\.name) == ["gemma4:4b", "gemma4:4b-q4_K_M"])
        #expect(models.map(\.size) == ["2.8 GB – 2.9 GB", "420 MB"])
    }

    @Test func readsInputCapabilityFromIndividualVariants() throws {
        let html = """
        <p>3 models</p>
        <span>vision</span>
        <a href="/library/model:one" class="md:hidden flex group">
          <span>0123456789ab</span> • 2.8GB • 128K context window •
          <span class="hidden sm:inline">
            Text, Image input • 1 month ago
          </span>
        </a>
        <a href="/library/model:two" class="md:hidden flex group"> • 2.9GB • Text input • 2 months ago</a>
        <a href="/library/model:three" class="md:hidden flex group">Missing input information</a>
        """
        let models = try OllamaCatalog.parseVariants(html, family: "model")
        #expect(models.map(\.inputCapability) == [.multimodal, .textOnly, nil])
    }

    @Test func distinguishesHostedOnlyFromAnUnreadablePage() throws {
        let html = """
        <p class="block col-span-6 md:hidden">3 models</p>
        <a href="/library/deepseek-v4.1-flash:cloud" class="md:hidden flex group">Medium Usage</a>
        <a href="/library/deepseek-v4.1-flash:1017-cloud" class="md:hidden flex group">Medium Usage</a>
        <a href="/library/deepseek-v4.1-flash:latest" class="md:hidden flex group">567966df89fe • Medium Usage • 1M context window</a>
        """
        #expect(try OllamaCatalog.parseVariants(html, family: "deepseek-v4.1-flash").isEmpty)
        #expect(throws: OllamaCatalog.Failure.unexpectedLayout) {
            try OllamaCatalog.parseVariants(html, family: "gemma4")
        }
        #expect(throws: OllamaCatalog.Failure.unexpectedLayout) {
            try OllamaCatalog.parseFamilies("<h1>The Ollama library is unavailable</h1>")
        }
    }
}
