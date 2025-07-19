/// A generic container holding a pair of values representing separated audio stems: vocals and instruments.
public struct Stems2<Value> {
    /// The value associated with the vocals stem.
    public let vocals: Value

    /// The value associated with the instruments stem.
    public let instruments: Value

    /// Initializes a new instance with given vocals and instruments values.
    ///
    /// - Parameters:
    ///   - vocals: The value for the vocals stem.
    ///   - instruments: The value for the instruments stem.
    public init(vocals: Value, instruments: Value) {
        self.vocals = vocals
        self.instruments = instruments
    }
}

extension Stems2 {
    /// Transforms both vocals and instruments values synchronously using a throwing closure.
    ///
    /// - Parameter transform: A closure that transforms a `Value` into another type.
    /// - Throws: Rethrows any error thrown by the transform closure.
    /// - Returns: A new `Stems2` instance with transformed values.
    public func mapStems<Transformed>(
        _ transform: (Value) throws -> Transformed
    ) rethrows -> Stems2<Transformed> {
        try Stems2<Transformed>(
            vocals: transform(vocals),
            instruments: transform(instruments)
        )
    }

    /// Transforms both vocals and instruments values asynchronously using a throwing closure.
    ///
    /// - Parameter transform: An async closure that transforms a `Value` into another type.
    /// - Throws: Rethrows any error thrown by the transform closure.
    /// - Returns: A new `Stems2` instance with asynchronously transformed values.
    public func asyncMapStems<Transformed>(
        _ transform: (Value) async throws -> Transformed
    ) async rethrows -> Stems2<Transformed> {
        try await Stems2<Transformed>(
            vocals: transform(vocals),
            instruments: transform(instruments)
        )
    }
}
