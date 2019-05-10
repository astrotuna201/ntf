//==============================================================================
/// QuantizeConverting
public protocol QuantizeConverting {
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

public extension QuantizeConverting {
    func convert(stored: Stored) -> Viewed { fatalError("not implemented") }
    func convert(viewed: Viewed) -> Stored { fatalError("not implemented") }
}

//==============================================================================
/// Integer --> Float
public extension QuantizeConverting where
    Stored: BinaryInteger, Viewed: BinaryFloatingPoint
{
    func convert(stored: Stored) -> Viewed {
        if stored == 0 {
            return 0
        } else if stored > 0 {
            return Viewed((Float(stored) + 1) * scale + bias)
        } else {
            return Viewed((Float(stored)) * scale + bias)
        }
    }
    
    func convert(viewed: Viewed) -> Stored {
        if viewed == 0 {
            return 0
        } else if viewed > 0 {
            return Stored(((Float(viewed) - bias) * oneOverScale) - 1)
        } else {
            return Stored((Float(viewed) - bias) * oneOverScale)
        }
    }
}

//==============================================================================
/// Float --> Integer -->
public extension QuantizeConverting where
    Stored: BinaryFloatingPoint, Viewed: BinaryInteger
{
    func convert(stored: Stored) -> Viewed {
        if stored == 0 {
            return 0
        } else if stored > 0 {
            return Viewed((Float(stored) - bias) * oneOverScale - 1)
        } else {
            return Viewed((Float(stored) - bias) * oneOverScale)
        }
    }
    
    func convert(viewed: Viewed) -> Stored {
        if viewed == 0 {
            return 0
        } else if viewed > 0 {
            return Stored(((Float(viewed) + 1 - bias) * scale))
        } else {
            return Stored((Float(viewed) - bias) * scale)
        }
    }
}

//==============================================================================
/// Quantizer
public struct Quantizer<Stored, Viewed>: QuantizeConverting {
    public let bias: Float
    public let scale: Float
    public let oneOverScale: Float
    
    public init(bias: Float, scale: Float) {
        self.bias = bias
        self.scale = scale
        self.oneOverScale = 1 / scale
    }
}
