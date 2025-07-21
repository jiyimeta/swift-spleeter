import Spleeter
import SwiftUI

enum SeparationStatus {
    case notStarted
    case processing(currentProgress: Int, total: Int, fraction: Float)
    case completed
    case error(any Error)
}

extension URL {
    static func temporaryDirectory(appending path: String) -> URL {
        FileManager.default.temporaryDirectory.appendingPathComponent(path)
    }
}

struct ContentView: View {
    @State private var status: SeparationStatus = .notStarted
    @State private var originalURL: URL?
    @State private var isDocumentPickerPresented = false
    private let vocalsURL: URL = .temporaryDirectory(appending: "vocals.wav")
    private let instrumentsURL: URL = .temporaryDirectory(appending: "instruments.wav")

    var body: some View {
        VStack {
            if let originalURL {
                AudioPlayerView(title: "Original", urls: [(name: " ", url: originalURL)])

                if #available(iOS 18.0, *) {
                    switch status {
                    case .notStarted:
                        Button("Start separation") {
                            predict(originalURL: originalURL)
                        }
                    case let .error(error):
                        VStack {
                            Button("Retry") {
                                predict(originalURL: originalURL)
                            }
                            Text(error.localizedDescription)
                                .font(.caption)
                                .foregroundStyle(.red)
                        }
                    case let .processing(currentProgress, total, fraction):
                        VStack {
                            ProgressView()
                            Text("\(currentProgress) / \(total) (\(fraction * 100, specifier: "%.0f")%)")
                        }
                    case .completed:
                        AudioPlayerView(
                            title: "Separated",
                            urls: [
                                (name: "Vocals", url: vocalsURL),
                                (name: "Instruments", url: instrumentsURL),
                            ]
                        )
                    }
                } else {
                    Text("Spleeter is unavailable.")
                        .foregroundStyle(.red)
                }
            } else {
                Button("Select file...") {
                    isDocumentPickerPresented = true
                }
                .sheet(isPresented: $isDocumentPickerPresented) {
                    DocumentPickerView(contentTypes: [.audio], fileURL: $originalURL)
                }
            }
        }
    }

    @available(iOS 18.0, *)
    private func predict(originalURL: URL) {
        Task {
            do {
                // swiftlint:disable:next force_unwrapping
                let modelURL = Bundle.main.url(forResource: "Spleeter2Model", withExtension: "mlmodelc")!
                let separator = try AudioSeparator2(modelURL: modelURL)
                for try await progress in separator.separate(
                    from: originalURL,
                    to: Stems2(vocals: vocalsURL, instruments: instrumentsURL)
                ) {
                    status = .processing(
                        currentProgress: progress.current,
                        total: progress.total,
                        fraction: progress.fraction
                    )
                }
                status = .completed
            } catch {
                status = .error(error)
            }
        }
    }
}
