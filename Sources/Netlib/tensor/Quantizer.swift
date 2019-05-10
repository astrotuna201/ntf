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
    
    @inlinable @inline(__always)
    func Float2Int<TF, TI>(value: TF) -> TI where TF: BinaryFloatingPoint, TI: BinaryInteger {
        if value > 0 {
            return TI(((Float(value) - bias) * oneOverScale) - 1)
        } else {
            return TI((Float(value) - bias) * oneOverScale)
        }
    }
}

//==============================================================================
/// Integer --> Float
/// NOTE: It's likely most of the time Viewed will be Float, so the cast of
/// Viewed(bias) should be thrown out by the compiler. In the case of
/// something like Float16, it will likely be faster to cast anyway.
public extension QuantizeConverting where
    Stored: BinaryInteger, Viewed: BinaryFloatingPoint
{
    @inlinable @inline(__always)
    func convert(stored: Stored) -> Viewed {
        if stored == 0 {
            return Viewed(bias)
        } else if stored > 0 {
            return Viewed((Float(stored) + 1) * scale + bias)
        } else {
            return Viewed((Float(stored)) * scale + bias)
        }
    }
    
    @inlinable @inline(__always)
    func convert(viewed: Viewed) -> Stored {
        if viewed == Viewed(bias) {
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
    @inlinable @inline(__always)
    func convert(stored: Stored) -> Viewed {
        if stored == 0 {
            return 0
        } else if stored > 0 {
            return Viewed((Float(stored) - bias) * oneOverScale - 1)
        } else {
            return Viewed((Float(stored) - bias) * oneOverScale)
        }
    }
    
    @inlinable @inline(__always)
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
    
    public init(scale: Float, bias: Float = 0) {
        self.bias = bias
        self.scale = scale
        self.oneOverScale = 1 / scale
    }
}
