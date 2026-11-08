extension Stems2 {
    /// The stems from `values`, in `values` order (`SpleeterStemLayout.two.stemNames`).
    public init(_ values: [Value]) {
        precondition(values.count == 2, "Stems2 takes 2 values")
        self.init(vocals: values[0], accompaniment: values[1])
    }
}

extension Stems4 {
    /// The stems from `values`, in `values` order (`SpleeterStemLayout.four.stemNames`).
    public init(_ values: [Value]) {
        precondition(values.count == 4, "Stems4 takes 4 values")
        self.init(vocals: values[0], drums: values[1], bass: values[2], other: values[3])
    }
}

extension Stems5 {
    /// The stems from `values`, in `values` order (`SpleeterStemLayout.five.stemNames`).
    public init(_ values: [Value]) {
        precondition(values.count == 5, "Stems5 takes 5 values")
        self.init(vocals: values[0], piano: values[1], drums: values[2], bass: values[3], other: values[4])
    }
}
