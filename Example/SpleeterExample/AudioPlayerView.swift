import AVFAudio
import SwiftUI

private enum MixedAudioPlayerError: Error {
    case noItems
}

private enum PlayerState {
    case player(_ players: [(name: String, player: AVAudioPlayer)])
    case error(any Error)
    case none
}

struct AudioPlayerView: View {
    let title: String
    let urls: [(name: String, url: URL)]

    @State private var playerState: PlayerState = .none

    @State private var timer = Timer.publish(every: 0.1, on: .main, in: .common).autoconnect()

    @State private var currentTime: TimeInterval = 0
    @State private var isPlaying = false

    var body: some View {
        HStack(alignment: .top) {
            Text(title)
                .lineLimit(1)
                .bold()
                .frame(width: 100)

            VStack(alignment: .leading) {
                HStack {
                    Button {
                        play()
                    } label: {
                        Image(systemName: "play.fill")
                    }
                    .disabled(isPlaying)

                    switch playerState {
                    case let .player(players):
                        Button {
                            isPlaying = false
                            pause(players: players.map(\.player))
                        } label: {
                            Image(systemName: "pause.fill")
                        }
                        .disabled(!isPlaying)

                        Slider(
                            value: Binding(
                                get: { currentTime },
                                set: { seek(players: players.map(\.player), to: $0) }
                            ),
                            in: 0 ... (players.map(\.player.duration).max() ?? 0)
                        )
                    case .error, .none:
                        EmptyView()
                    }
                }

                switch playerState {
                case let .player(players):
                    ForEach(players, id: \.name) { player in
                        AudioChannelView(
                            name: player.name,
                            setVolume: {
                                player.player.setVolume($0, fadeDuration: 0.02)
                            }
                        )
                    }
                case let .error(error):
                    Text(error.localizedDescription)
                        .font(.caption)
                        .foregroundStyle(.red)
                case .none:
                    EmptyView()
                }
            }

            Spacer(minLength: 0)
        }
        .padding(8)
        .onReceive(timer) { _ in
            guard case let .player(players) = playerState else {
                return
            }

            currentTime = players[0].player.currentTime
            isPlaying = players[0].player.isPlaying
        }
    }

    private func play() {
        let players = switch playerState {
        case let .player(existingPlayers):
            players = existingPlayers.map(\.player)
        case .error, .none:
            do {
                let namedPlayers = try urls.map {
                    try (name: $0.name, player: AVAudioPlayer(contentsOf: $0.url))
                }
                players = namedPlayers.map(\.player)
                playerState = .player(namedPlayers)
            } catch {
                playerState = .error(error)
                return
            }
        }

        isPlaying = true

        for player in players {
            player.play()
        }
    }

    private func pause(players: [AVAudioPlayer]) {
        for player in players {
            player.pause()
        }
    }

    private func seek(players: [AVAudioPlayer], to time: TimeInterval) {
        for player in players {
            player.currentTime = time
        }
    }
}

struct AudioChannelView: View {
    let name: String
    let setVolume: (Float) -> Void

    @State private var isMuted = false
    @State private var volume: Float = 1

    var body: some View {
        HStack {
            Text(name)
                .frame(width: 100)

            Button {
                isMuted.toggle()
            } label: {
                if isMuted {
                    Image(systemName: "speaker.slash.fill")
                } else {
                    Image(systemName: "speaker.wave.2.fill")
                }
            }

            Slider(
                value: $volume,
                in: 0 ... 1
            )
            .disabled(isMuted)
        }
        .onChange(of: isMuted) { _, newValue in
            setVolume(newValue ? 0 : volume)
        }
        .onChange(of: volume) { _, newValue in
            setVolume(newValue)
        }
    }
}
