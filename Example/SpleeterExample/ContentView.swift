import Spleeter
import SwiftUI

enum SeparationStatus {
    case notStarted
    case processing(currentProgress: Int, total: Int, fraction: Float)
    case completed(stems: Stems)
    case error(any Error)
}

enum Stems {
    case stems2(Stems2<NamedURL>)
    case stems4(Stems4<NamedURL>)
    case stems5(Stems5<NamedURL>)

    var values: [NamedURL] {
        switch self {
        case let .stems2(stems2): stems2.values
        case let .stems4(stems4): stems4.values
        case let .stems5(stems5): stems5.values
        }
    }
}

struct NamedURL {
    let name: String
    let url: URL
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

    var body: some View {
        VStack {
            if let originalURL {
                AudioPlayerView(title: "Original", namedURLs: [NamedURL(name: " ", url: originalURL)])

                if #available(iOS 18.0, *) {
                    switch status {
                    case .notStarted:
                        separationButtons(originalURL: originalURL)
                    case let .error(error):
                        VStack {
                            separationButtons(originalURL: originalURL)
                            Text(error.localizedDescription)
                                .font(.caption)
                                .foregroundStyle(.red)
                        }
                    case let .processing(currentProgress, total, fraction):
                        VStack {
                            ProgressView()
                            Text("\(currentProgress) / \(total) (\(fraction * 100, specifier: "%.0f")%)")
                        }
                    case let .completed(stems):
                        AudioPlayerView(
                            title: "Separated",
                            namedURLs: stems.values
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

        Button("Reset") {
            status = .notStarted
            originalURL = nil
        }
    }

    @available(iOS 18.0, *)
    private func predict(originalURL: URL) {
        Task {
            let urls = Stems2<URL>(
                vocals: .temporaryDirectory(appending: "vocals.wav"),
                accompaniment: .temporaryDirectory(appending: "accompaniment.wav")
            )
            do {
                // swiftlint:disable:next force_unwrapping
                let modelURL = Bundle.main.url(forResource: "Spleeter2Model", withExtension: "mlmodelc")!
                let separator = try AudioSeparator2(modelURL: modelURL)
                for try await progress in separator.separate(from: originalURL, to: urls) {
                    status = .processing(
                        currentProgress: progress.current,
                        total: progress.total,
                        fraction: progress.fraction
                    )
                }
                status = .completed(
                    stems: .stems2(
                        Stems2(
                            vocals: NamedURL(name: "Vocals", url: urls.vocals),
                            accompaniment: NamedURL(name: "Accompaniment", url: urls.accompaniment)
                        )
                    )
                )
            } catch {
                status = .error(error)
            }
        }
    }

    @available(iOS 18.0, *)
    @ViewBuilder
    private func separationButtons(originalURL: URL) -> some View {
        Button("Start 2-stem separation") {
            Task {
                let urls = Stems2<URL>(
                    vocals: .temporaryDirectory(appending: "vocals.wav"),
                    accompaniment: .temporaryDirectory(appending: "accompaniment.wav")
                )
                do {
                    // swiftlint:disable:next force_unwrapping
                    let modelURL = Bundle.main.url(forResource: "Spleeter2Model", withExtension: "mlmodelc")!
                    let separator = try AudioSeparator2(modelURL: modelURL)
                    for try await progress in separator.separate(from: originalURL, to: urls) {
                        status = .processing(
                            currentProgress: progress.current,
                            total: progress.total,
                            fraction: progress.fraction
                        )
                    }
                    status = .completed(
                        stems: .stems2(
                            Stems2(
                                vocals: NamedURL(name: "Vocals", url: urls.vocals),
                                accompaniment: NamedURL(name: "Accompaniment", url: urls.accompaniment)
                            )
                        )
                    )
                } catch {
                    status = .error(error)
                }
            }
        }

        Button("Start 4-stem separation") {
            Task {
                let urls = Stems4<URL>(
                    vocals: .temporaryDirectory(appending: "vocals.wav"),
                    drums: .temporaryDirectory(appending: "drums.wav"),
                    bass: .temporaryDirectory(appending: "bass.wav"),
                    other: .temporaryDirectory(appending: "other.wav")
                )
                do {
                    // swiftlint:disable:next force_unwrapping
                    let modelURL = Bundle.main.url(forResource: "Spleeter4Model", withExtension: "mlmodelc")!
                    let separator = try AudioSeparator4(modelURL: modelURL)
                    for try await progress in separator.separate(from: originalURL, to: urls) {
                        status = .processing(
                            currentProgress: progress.current,
                            total: progress.total,
                            fraction: progress.fraction
                        )
                    }
                    status = .completed(
                        stems: .stems4(
                            Stems4(
                                vocals: NamedURL(name: "Vocals", url: urls.vocals),
                                drums: NamedURL(name: "Drums", url: urls.drums),
                                bass: NamedURL(name: "Bass", url: urls.bass),
                                other: NamedURL(name: "Other", url: urls.other)
                            )
                        )
                    )
                } catch {
                    status = .error(error)
                }
            }
        }

        Button("Start 5-stem separation") {
            Task {
                let urls = Stems5<URL>(
                    vocals: .temporaryDirectory(appending: "vocals.wav"),
                    piano: .temporaryDirectory(appending: "piano.wav"),
                    drums: .temporaryDirectory(appending: "drums.wav"),
                    bass: .temporaryDirectory(appending: "bass.wav"),
                    other: .temporaryDirectory(appending: "other.wav")
                )
                do {
                    // swiftlint:disable:next force_unwrapping
                    let modelURL = Bundle.main.url(forResource: "Spleeter5Model", withExtension: "mlmodelc")!
                    let separator = try AudioSeparator5(modelURL: modelURL)
                    for try await progress in separator.separate(from: originalURL, to: urls) {
                        status = .processing(
                            currentProgress: progress.current,
                            total: progress.total,
                            fraction: progress.fraction
                        )
                    }
                    status = .completed(
                        stems: .stems5(
                            Stems5(
                                vocals: NamedURL(name: "Vocals", url: urls.vocals),
                                piano: NamedURL(name: "Piano", url: urls.piano),
                                drums: NamedURL(name: "Drums", url: urls.drums),
                                bass: NamedURL(name: "Bass", url: urls.bass),
                                other: NamedURL(name: "Other", url: urls.other)
                            )
                        )
                    )
                } catch {
                    status = .error(error)
                }
            }
        }
    }

    enum Error: Swift.Error {
        case failedToReachFile
    }
}
