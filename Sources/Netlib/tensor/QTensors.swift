//==============================================================================
/// Quantizer
public protocol Quantizer {
    associatedtype Stored
    associatedtype Viewed
    
    /// the bias to apply during conversion
    var bias: Float { get }
    /// the scale to apply during conversion
    var scale: Float { get }
    var oneOverScale: Float { get }
    /// converts from Scalar to ViewedScalar
    func convert(stored: Stored) -> Viewed
    /// converts from Scalar to ViewedScalar
    func convert(viewed: Viewed) -> Stored
}

public extension Quantizer {
    func convert(stored: Stored) -> Viewed { fatalError("not implemented") }
    func convert(viewed: Viewed) -> Stored { fatalError("not implemented") }
}

public extension Quantizer where Stored: BinaryInteger, Viewed: BinaryFloatingPoint {
    func convert(stored: Stored) -> Viewed {
        let value = stored == 0 ? 0 : (Float(stored) + 1) * scale + bias
        return Viewed(value)
    }
    
    func convert(viewed: Viewed) -> Stored {
        let value: Float = viewed == 0 ? 0 : ((Float(viewed) - bias) * oneOverScale) - 1
        return Stored(value)
    }
}

public extension Quantizer where Stored: BinaryFloatingPoint, Viewed: BinaryInteger {
    func convert(stored: Stored) -> Viewed {
        let value = stored == 0 ? 0 : Float(stored) * scale + bias
        return Viewed(value)
    }
    
    func convert(viewed: Viewed) -> Stored {
        return Stored((Float(viewed) - bias) * oneOverScale)
    }
}

public extension Quantizer where Stored: BinaryInteger, Viewed: BinaryInteger {
    func convert(stored: Stored) -> Viewed {
        let value = stored == 0 ? 0 : (Float(stored) + 1) * scale + bias
        return Viewed(value)
    }
    
    func convert(viewed: Viewed) -> Stored {
        return Stored(Float(viewed) / scale + bias)
    }
}


public struct QConverter<Stored, Viewed>: Quantizer {
    public let bias: Float
    public let scale: Float
    public let oneOverScale: Float
    
    public init(bias: Float, scale: Float) {
        self.bias = bias
        self.scale = scale
        self.oneOverScale = 1 / scale
    }
}
