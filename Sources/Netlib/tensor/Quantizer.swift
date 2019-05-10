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

//==============================================================================
/// QuantizeConverting
public extension QuantizeConverting {
    func convert(stored: Stored) -> Viewed { fatalError("not implemented") }
    func convert(viewed: Viewed) -> Stored { fatalError("not implemented") }
    
    //==========================================================================
    /// NOTE: It's likely most of the time value will be Float, so the cast of
    /// TF(bias) should be thrown out by the compiler. In the case of
    /// something like Float16, it will likely be faster to cast anyway.
    @inlinable @inline(__always)
    func Float2Int<TF, TI>(value: TF) -> TI
        where TF: BinaryFloatingPoint, TI: BinaryInteger
    {
        if value == TF(bias) {
            return 0
        } else if value > 0 {
            return TI(((Float(value) - bias) * oneOverScale) - 1)
        } else {
            return TI((Float(value) - bias) * oneOverScale)
        }
    }

    //==========================================================================
    @inlinable @inline(__always)
    func Int2Float<TI, TF>(value: TI) -> TF
        where TI: BinaryInteger, TF: BinaryFloatingPoint
    {
        if value == 0 {
            return TF(bias)
        } else if value > 0 {
            return TF((Float(value) + 1) * scale + bias)
        } else {
            return TF((Float(value)) * scale + bias)
        }
    }
}

//==============================================================================
/// Integer --> Float
public extension QuantizeConverting where
    Stored: BinaryInteger, Viewed: BinaryFloatingPoint
{
    @inlinable @inline(__always)
    func convert(stored: Stored) -> Viewed {
        return Int2Float(value: stored)
    }
    
    @inlinable @inline(__always)
    func convert(viewed: Viewed) -> Stored {
        return Float2Int(value: viewed)
    }
}

//==============================================================================
/// Float --> Integer -->
public extension QuantizeConverting where
    Stored: BinaryFloatingPoint, Viewed: BinaryInteger
{
    @inlinable @inline(__always)
    func convert(stored: Stored) -> Viewed {
        return Float2Int(value: stored)
    }
    
    @inlinable @inline(__always)
    func convert(viewed: Viewed) -> Stored {
        return Int2Float(value: viewed)
    }
}

//==============================================================================
/// Integer --> Float
//public extension QuantizeConverting where
//    Stored: BinaryInteger,
//    Viewed: BinaryFloatingPoint
//{
//    @inlinable @inline(__always)
//    func convert(stored: Stored) -> Viewed {
//        return Int2Float(value: stored)
//    }
//
//    @inlinable @inline(__always)
//    func convert(viewed: Viewed) -> Stored {
//        return Float2Int(value: viewed)
//    }
//}

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
